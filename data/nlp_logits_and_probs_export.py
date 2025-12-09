#!/usr/bin/env python3
"""
Export pre-softmax logits and post-softmax probabilities for NLP models
with the SAME on-disk format as your vision exporter:

- Per-vector header '<IHH' (uint32 length, uint16 source_id, uint16 flags=0)
- Then `length` float32 values (contiguous)
- Two streams per run: *_logits.bin and *_probs.bin (paired order)
- A compact manifest.json with counts and SHA-256 checksums

Supported tasks (public weights + public datasets):
  • mBERT Masked-LM  (model='bert-base-multilingual-cased') on XNLI (validation+test)
  • GPT-2 Causal LM  (model='gpt2') on WikiText-103 test

USAGE EXAMPLES
--------------
# 1) mBERT (MLM) only, multilingual (default langs)
python nlp_logits_and_probs_export.py \
  --task mbert_mlm --outdir ./exports/nlp_mlm \
  --mlm-max-vectors 6000 --mlm-langs en,de,fr,es,zh --mlm-max-seq 256

# 2) GPT-2 (CLM) only
python nlp_logits_and_probs_export.py \
  --task gpt2_clm --outdir ./exports/nlp_clm \
  --clm-max-vectors 8000 --clm-stride 8 --clm-max-seq 512

# 3) Both in one go
python nlp_logits_and_probs_export.py \
  --task both --outdir ./exports/nlp_both \
  --mlm-max-vectors 6000 --mlm-langs en,de,fr,es,zh --mlm-max-seq 256 \
  --clm-max-vectors 8000 --clm-stride 8 --clm-max-seq 512

Dependencies: pip install transformers datasets torch
"""

import argparse
import json
import os
import struct
import hashlib
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

# Optional but recommended: datasets/transformers
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
except Exception as e:
    raise SystemExit("This script requires 'transformers' and 'datasets'.\nInstall via: pip install transformers datasets torch\n" + str(e))

# -------------------------------
# Binary format (identical to vision exporter)
# -------------------------------
HEADER_FMT = '<IHH'  # uint32 length, uint16 source_id, uint16 flags
HEADER_SIZE = struct.calcsize(HEADER_FMT)

SOURCE_ID = {
    'mbert_mlm': 1001,
    'gpt2_clm':  2001,
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_record(fh, vec: torch.Tensor, source_id: int, flags: int = 0):
    assert vec.ndim == 1, f"Expected 1D vector, got shape {tuple(vec.shape)}"
    vec = vec.to(torch.float32).contiguous().cpu()
    length = int(vec.numel())
    fh.write(struct.pack(HEADER_FMT, length, source_id, flags))
    fh.write(vec.numpy().tobytes(order='C'))


# -------------------------------
# Timing utilities for softmax monitoring
# -------------------------------

class SoftmaxTimer:
    """Track timing for softmax operations, both explicit and within model."""
    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda_events = (device.type == 'cuda')
        self.final_softmax_times: List[float] = []
        self.model_softmax_times: List[float] = []
        self.model_softmax_locations: List[str] = []
        self.forward_pass_times: List[float] = []  # Track total forward pass time
        self.in_forward_pass = False  # Flag to track if we're in forward pass
        
        if self.use_cuda_events:
            self.final_start_event = torch.cuda.Event(enable_timing=True)
            self.final_end_event = torch.cuda.Event(enable_timing=True)
            self.forward_start_event = torch.cuda.Event(enable_timing=True)
            self.forward_end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.final_start_time = None
            self.forward_start_time = None
            
    def start_final_softmax(self):
        """Mark start of explicit softmax on final logits."""
        if self.use_cuda_events:
            self.final_start_event.record()
        else:
            self.final_start_time = time.perf_counter()
            
    def end_final_softmax(self):
        """Mark end of explicit softmax on final logits."""
        if self.use_cuda_events:
            self.final_end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.final_start_event.elapsed_time(self.final_end_event)
            self.final_softmax_times.append(elapsed_ms / 1000.0)  # Convert to seconds
        else:
            if self.final_start_time is not None:
                elapsed = time.perf_counter() - self.final_start_time
                self.final_softmax_times.append(elapsed)
    
    def start_forward_pass(self):
        """Mark start of model forward pass."""
        self.in_forward_pass = True
        if self.use_cuda_events:
            torch.cuda.synchronize()  # Ensure previous ops are done
            self.forward_start_event.record()
        else:
            self.forward_start_time = time.perf_counter()
    
    def end_forward_pass(self):
        """Mark end of model forward pass."""
        self.in_forward_pass = False
        if self.use_cuda_events:
            self.forward_end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.forward_start_event.elapsed_time(self.forward_end_event)
            self.forward_pass_times.append(elapsed_ms / 1000.0)  # Convert to seconds
        else:
            if self.forward_start_time is not None:
                elapsed = time.perf_counter() - self.forward_start_time
                self.forward_pass_times.append(elapsed)
                
    def record_model_softmax(self, location: str, elapsed_seconds: float):
        """Record a softmax operation from within the model (only if in forward pass)."""
        if self.in_forward_pass:
            self.model_softmax_times.append(elapsed_seconds)
            self.model_softmax_locations.append(location)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics including portion calculations."""
        # Calculate totals
        total_forward_time = sum(self.forward_pass_times) if self.forward_pass_times else 0.0
        total_model_softmax_time = sum(self.model_softmax_times) if self.model_softmax_times else 0.0
        total_final_softmax_time = sum(self.final_softmax_times) if self.final_softmax_times else 0.0
        total_all_softmax_time = total_model_softmax_time + total_final_softmax_time
        
        # Calculate portions (percentages)
        softmax_portion_in_forward = 0.0
        all_softmax_portion_in_forward = 0.0
        if total_forward_time > 0:
            softmax_portion_in_forward = (total_model_softmax_time / total_forward_time) * 100.0
            all_softmax_portion_in_forward = (total_all_softmax_time / total_forward_time) * 100.0
        
        stats = {
            'forward_pass': {
                'count': len(self.forward_pass_times),
                'total_seconds': total_forward_time,
                'mean_seconds': total_forward_time / len(self.forward_pass_times) if self.forward_pass_times else 0.0,
                'min_seconds': min(self.forward_pass_times) if self.forward_pass_times else 0.0,
                'max_seconds': max(self.forward_pass_times) if self.forward_pass_times else 0.0,
            },
            'model_softmax': {
                'count': len(self.model_softmax_times),
                'total_seconds': total_model_softmax_time,
                'mean_seconds': total_model_softmax_time / len(self.model_softmax_times) if self.model_softmax_times else 0.0,
                'min_seconds': min(self.model_softmax_times) if self.model_softmax_times else 0.0,
                'max_seconds': max(self.model_softmax_times) if self.model_softmax_times else 0.0,
            },
            'final_softmax': {
                'count': len(self.final_softmax_times),
                'total_seconds': total_final_softmax_time,
                'mean_seconds': total_final_softmax_time / len(self.final_softmax_times) if self.final_softmax_times else 0.0,
                'min_seconds': min(self.final_softmax_times) if self.final_softmax_times else 0.0,
                'max_seconds': max(self.final_softmax_times) if self.final_softmax_times else 0.0,
            },
            'portions': {
                'softmax_in_forward_percent': softmax_portion_in_forward,
                'all_softmax_in_forward_percent': all_softmax_portion_in_forward,
                'model_softmax_time': total_model_softmax_time,
                'forward_pass_time': total_forward_time,
                'ratio_softmax_to_forward': total_model_softmax_time / total_forward_time if total_forward_time > 0 else 0.0,
            },
        }
        
        # Group by location for model softmax
        if self.model_softmax_locations:
            location_counts = defaultdict(int)
            location_times = defaultdict(list)
            for loc, elapsed in zip(self.model_softmax_locations, self.model_softmax_times):
                location_counts[loc] += 1
                location_times[loc].append(elapsed)
            
            stats['model_softmax']['by_location'] = {
                loc: {
                    'count': location_counts[loc],
                    'total_seconds': sum(location_times[loc]),
                    'mean_seconds': sum(location_times[loc]) / len(location_times[loc]),
                    'portion_percent': (sum(location_times[loc]) / total_forward_time * 100.0) if total_forward_time > 0 else 0.0,
                }
                for loc in location_counts
            }
        
        return stats


class BoundMethodWrapper:
    """Helper to create a bound method wrapper."""
    def __init__(self, wrapper, instance):
        self.wrapper = wrapper
        self.instance = instance
    
    def __call__(self, *args, **kwargs):
        return self.wrapper(self.instance, *args, **kwargs)


class TimedSoftmaxWrapper:
    """Wrapper to track all softmax calls, including those within the model."""
    def __init__(self, timer: SoftmaxTimer, original_softmax, name="F.softmax"):
        self.timer = timer
        self.original_softmax = original_softmax
        self.use_cuda = timer.use_cuda_events
        self.call_count = 0
        self.name = name
    
    def __get__(self, obj, objtype=None):
        """Descriptor protocol: allows this to work as a bound method when assigned to a class."""
        if obj is None:
            return self
        return BoundMethodWrapper(self, obj)
        
    def __call__(self, *args, **kwargs):
        # Get caller location for identification
        import inspect
        location = "unknown"
        module_context = ""
        
        # Walk up the call stack to find useful context
        frame = inspect.currentframe().f_back
        stack_depth = 0
        max_depth = 5
        
        while frame and stack_depth < max_depth:
            try:
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                func_name = frame.f_code.co_name
                
                # Check if this frame is in an Attention module or related
                if 'attention' in filename.lower() or 'attention' in func_name.lower():
                    module_context = f"[Attention/{func_name}]"
                    break
                
                # Check for transformers-specific paths
                if 'transformers' in filename.lower():
                    if 'attention' in func_name.lower() or 'attn' in func_name.lower():
                        module_context = f"[transformers.Attention/{func_name}]"
                        break
                
                # Use the first frame's location as fallback
                if stack_depth == 0:
                    location = f"{Path(filename).name}:{lineno}"
                
                frame = frame.f_back
                stack_depth += 1
            except:
                break
        
        # Create location string
        if module_context:
            location = f"{module_context} {location}"
        elif location == "unknown":
            location = f"{self.name}:{stack_depth}"
        
        self.call_count += 1
        
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            result = self.original_softmax(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed_seconds = elapsed_ms / 1000.0
        else:
            start_time = time.perf_counter()
            result = self.original_softmax(*args, **kwargs)
            elapsed_seconds = time.perf_counter() - start_time
        
        # Record all softmax calls (will filter later if needed)
        self.timer.record_model_softmax(location, elapsed_seconds)
        
        return result


def wrap_softmax_for_timing(timer: SoftmaxTimer):
    """Wrap all softmax variants to track timing of all softmax calls."""
    original_F_softmax = F.softmax
    original_torch_softmax = torch.softmax
    original_tensor_softmax = torch.Tensor.softmax
    
    wrapper_F = TimedSoftmaxWrapper(timer, original_F_softmax, name="F.softmax")
    wrapper_torch = TimedSoftmaxWrapper(timer, original_torch_softmax, name="torch.softmax")
    wrapper_tensor = TimedSoftmaxWrapper(timer, original_tensor_softmax, name="Tensor.softmax")
    
    F.softmax = wrapper_F
    torch.softmax = wrapper_torch
    torch.Tensor.softmax = wrapper_tensor
    torch.nn.functional.softmax = wrapper_F
    
    return (original_F_softmax, original_torch_softmax, original_tensor_softmax)


def restore_softmax(originals):
    """Restore original softmax functions after timing."""
    original_F_softmax, original_torch_softmax, original_tensor_softmax = originals
    F.softmax = original_F_softmax
    torch.nn.functional.softmax = original_F_softmax
    torch.softmax = original_torch_softmax
    torch.Tensor.softmax = original_tensor_softmax


# -------------------------------
# Timing summary helper
# -------------------------------

def _print_timing_summary(task_name: str, timing_stats: Dict[str, Any]):
    """Print timing summary for a task."""
    print(f"\n==== SOFTMAX TIMING SUMMARY: {task_name} ====")
    
    # Forward pass timing
    forward_stats = timing_stats['forward_pass']
    print(f"Forward Pass (model inference):")
    print(f"  Count: {forward_stats['count']}")
    print(f"  Total time: {forward_stats['total_seconds']:.4f}s")
    print(f"  Mean per sample: {forward_stats['mean_seconds']*1000:.4f}ms")
    print(f"  Min: {forward_stats['min_seconds']*1000:.4f}ms, Max: {forward_stats['max_seconds']*1000:.4f}ms")
    
    # Model softmax (within forward pass)
    model_stats = timing_stats['model_softmax']
    print(f"\nModel Softmax (in architecture - attention, etc.):")
    print(f"  Total calls: {model_stats['count']}")
    print(f"  Total time: {model_stats['total_seconds']:.4f}s")
    
    # Check if attention softmax was detected
    attention_detected = False
    if 'by_location' in model_stats and model_stats['by_location']:
        for loc in model_stats['by_location'].keys():
            if 'attention' in loc.lower() or 'attn' in loc.lower() or 'transformers' in loc.lower():
                attention_detected = True
                break
    
    if model_stats['count'] > 0:
        if attention_detected:
            print(f"  ✓ Attention softmax operations detected and timed")
        else:
            print(f"  ✓ Softmax operations detected (may include attention)")
        print(f"  Mean per call: {model_stats['mean_seconds']*1000:.4f}ms")
        print(f"  Min: {model_stats['min_seconds']*1000:.4f}ms, Max: {model_stats['max_seconds']*1000:.4f}ms")
        
        if 'by_location' in model_stats and model_stats['by_location']:
            print(f"\n  By location (top 10):")
            sorted_locs = sorted(model_stats['by_location'].items(), 
                               key=lambda x: x[1]['total_seconds'], reverse=True)
            for loc, stats in sorted_locs[:10]:
                print(f"    {loc}: {stats['count']} calls, {stats['total_seconds']*1000:.4f}ms total "
                      f"({stats['portion_percent']:.2f}% of forward), {stats['mean_seconds']*1000:.4f}ms avg")
            if len(sorted_locs) > 10:
                print(f"    ... and {len(sorted_locs) - 10} more locations")
    else:
        print(f"  ⚠ Warning: No softmax operations detected during forward pass!")
        print(f"     This may indicate that softmax is not being called via:")
        print(f"     - F.softmax / nn.functional.softmax")
        print(f"     - torch.softmax")
        print(f"     - tensor.softmax()")
        print(f"     or that the model uses a custom implementation.")
    
    # Portion calculation
    portions = timing_stats['portions']
    print(f"\n==== SOFTMAX PORTION IN FORWARD PASS ====")
    print(f"Forward pass total time: {portions['forward_pass_time']:.4f}s")
    print(f"Softmax time in forward pass: {portions['model_softmax_time']:.4f}s")
    print(f"Portion: {portions['softmax_in_forward_percent']:.2f}% "
          f"({portions['ratio_softmax_to_forward']:.4f}x)")
    print(f"\nThis means softmax operations account for {portions['softmax_in_forward_percent']:.2f}% "
          f"of the forward pass time.")
    
    # Final softmax (post-forward)
    final_stats = timing_stats['final_softmax']
    print(f"\nFinal Softmax (on logits, post-forward):")
    print(f"  Count: {final_stats['count']}")
    print(f"  Total time: {final_stats['total_seconds']:.4f}s")
    print(f"  Mean per sample: {final_stats['mean_seconds']*1000:.4f}ms")
    print(f"  Min: {final_stats['min_seconds']*1000:.4f}ms, Max: {final_stats['max_seconds']*1000:.4f}ms")
    
    if portions['all_softmax_in_forward_percent'] > 0:
        print(f"\nIncluding final softmax: {portions['all_softmax_in_forward_percent']:.2f}% of forward pass")


# -------------------------------
# Explicit forward passes for NLP models
# -------------------------------

def explicit_forward_bert_mlm(model, input_ids, attention_mask=None):
    """Explicit forward pass for BERT MLM model.
    
    Manually calls each component to expose all softmax operations:
    - Embeddings (token, position, token_type)
    - Transformer encoder layers (each with attention containing softmax)
    - Final layer norm
    - MLM head
    """
    # 1. Embeddings
    embeddings = model.bert.embeddings
    x = embeddings.word_embeddings(input_ids)  # [B, T, hidden_size]
    # Position embeddings - handle both parameter and nn.Embedding cases
    if hasattr(embeddings.position_embeddings, 'weight'):
        pos_emb = embeddings.position_embeddings.weight[:x.size(1), :].unsqueeze(0)
    else:
        position_ids = torch.arange(x.size(1), device=input_ids.device).unsqueeze(0)
        pos_emb = embeddings.position_embeddings(position_ids)
    x = x + pos_emb
    if hasattr(embeddings, 'token_type_embeddings') and embeddings.token_type_embeddings is not None:
        # For BERT, token_type_ids are typically all zeros for single sequences
        token_type_ids = torch.zeros_like(input_ids)
        x = x + embeddings.token_type_embeddings(token_type_ids)
    x = embeddings.LayerNorm(x)
    x = embeddings.dropout(x)
    
    # 2. Transformer encoder layers (each layer contains attention with softmax)
    encoder = model.bert.encoder
    for layer in encoder.layer:
        # BERT layer structure:
        # - attention (with explicit softmax) -> add & norm
        # - feedforward -> add & norm
        
        # Attention branch
        residual = x
        
        # Get attention modules
        attn_module = layer.attention.self  # BertSelfAttention
        attn_output = layer.attention.output  # BertSelfOutput
        
        # Compute Q, K, V
        B, T, C = x.shape
        q = attn_module.query(x)  # [B, T, hidden_size]
        k = attn_module.key(x)    # [B, T, hidden_size]
        v = attn_module.value(x)  # [B, T, hidden_size]
        
        # Get number of heads from config or module
        if hasattr(attn_module, 'num_attention_heads'):
            num_heads = attn_module.num_attention_heads
        elif hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
        else:
            # Default for BERT-base: 12 heads
            num_heads = 12
        
        head_dim = C // num_heads

        
        # Reshape Q, K, V: [B, T, hidden_size] -> [B, num_heads, T, head_dim]
        q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, T, T]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, 1, T] for broadcasting
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, T]
            attn_mask = (1.0 - attn_mask) * -10000.0  # Mask positions become -inf
            attn_scores = attn_scores + attn_mask
        
        # EXPLICIT SOFTMAX: This is what we want to time explicitly!
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, T, T]
        
        # Apply attention dropout (if present)
        if hasattr(attn_module, 'dropout'):
            attn_weights = attn_module.dropout(attn_weights)
        
        # Apply attention to values
        x_attn = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]
        
        # Reshape back: [B, num_heads, T, head_dim] -> [B, T, hidden_size]
        x_attn = x_attn.transpose(1, 2).reshape(B, T, C)  # [B, T, hidden_size]
        
        # Apply output projection and dropout
        x_attn = attn_output.dense(x_attn)  # [B, T, hidden_size]
        x_attn = attn_output.LayerNorm(x_attn + residual)  # Add residual and norm
        x_attn = attn_output.dropout(x_attn)
        x = x_attn
        
        # Feedforward branch
        residual = x
        x = layer.intermediate.dense(x)  # [B, T, intermediate_size]
        x = layer.intermediate.intermediate_act_fn(x)  # GELU activation
        x = layer.output.dense(x)  # [B, T, hidden_size]
        x = layer.output.dropout(x)
        x = layer.output.LayerNorm(x + residual)  # Add residual and norm
    
    # 3. MLM head (no final layer norm needed - last layer output is already normalized)
    # The transform module applies: dense -> activation -> LayerNorm
    transform = model.cls.predictions.transform
    x = transform.dense(x)  # [B, T, hidden_size]
    # Apply activation - BERT uses GELU (can be accessed via config.hidden_act or use F.gelu)
    # Use F.gelu directly as it's the standard activation for BERT
    x = F.gelu(x)
    x = transform.LayerNorm(x)
    output = model.cls.predictions.decoder(x)  # [B, T, vocab_size]
    return output


def explicit_forward_gpt2_clm(model, input_ids):
    """Explicit forward pass for GPT-2 CLM model.
    
    Manually calls each component to expose all softmax operations:
    - Token and position embeddings
    - Transformer decoder layers (each with masked attention containing softmax)
    - Final layer norm
    - LM head
    """
    # 1. Embeddings
    embeddings = model.transformer.wte  # Word token embeddings
    x = embeddings(input_ids)  # [B, T, hidden_size]
    
    # Position embeddings
    if hasattr(model.transformer, 'wpe'):
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = x + model.transformer.wpe(position_ids)  # [B, T, hidden_size]
    
    # Dropout
    x = model.transformer.drop(x)
    
    # 2. Transformer decoder blocks (each block contains masked attention with softmax)
    for block in model.transformer.h:
        # GPT-2 block structure:
        # - ln_1 -> attention (with explicit softmax and causal mask) -> add
        # - ln_2 -> feedforward -> add
        
        # Attention branch
        residual = x
        x = block.ln_1(x)  # Layer norm before attention
        
        # Get attention module
        attn = block.attn
        
        # Compute Q, K, V (GPT-2 uses combined QKV projection)
        B, T, C = x.shape
        
        # GPT-2 uses c_attn which projects to 3*C
        qkv = attn.c_attn(x)  # [B, T, 3*hidden_size]
        # Split into Q, K, V
        qkv = qkv.split(C, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, T, hidden_size]
        
        # Get number of heads
        if hasattr(attn, 'num_heads'):
            num_heads = attn.num_heads
        elif hasattr(model, 'config') and hasattr(model.config, 'n_head'):
            num_heads = model.config.n_head
        else:
            # Default for GPT-2: 12 heads
            num_heads = 12
        
        head_dim = C // num_heads
        
        # Reshape Q, K, V: [B, T, hidden_size] -> [B, num_heads, T, head_dim]
        q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, T, T]
        
        # Apply causal mask (lower triangular mask)
        # Create causal mask: positions that should be masked get -inf
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # EXPLICIT SOFTMAX: This is what we want to time explicitly!
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, T, T]
        
        # Apply attention dropout (if present)
        if hasattr(attn, 'attn_dropout'):
            attn_weights = attn.attn_dropout(attn_weights)
        elif hasattr(attn, 'dropout'):
            attn_weights = attn.dropout(attn_weights)
        
        # Apply attention to values
        x_attn = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]
        
        # Reshape back: [B, num_heads, T, head_dim] -> [B, T, hidden_size]
        x_attn = x_attn.transpose(1, 2).reshape(B, T, C)  # [B, T, hidden_size]
        
        # Apply output projection and dropout
        x_attn = attn.c_proj(x_attn)  # [B, T, hidden_size]
        if hasattr(attn, 'resid_dropout'):
            x_attn = attn.resid_dropout(x_attn)
        elif hasattr(attn, 'dropout'):
            x_attn = attn.dropout(x_attn)
        
        # Add residual connection
        x = x_attn + residual
        
        # Feedforward branch
        residual = x
        x = block.ln_2(x)  # Layer norm before feedforward
        x = block.mlp.c_fc(x)  # [B, T, intermediate_size]
        x = block.mlp.act(x)  # GELU activation
        x = block.mlp.c_proj(x)  # [B, T, hidden_size]
        if hasattr(block.mlp, 'dropout'):
            x = block.mlp.dropout(x)
        x = x + residual  # Add residual connection
    
    # 3. Final layer norm
    x = model.transformer.ln_f(x)  # [B, T, hidden_size]
    
    # 4. LM head
    output = model.lm_head(x)  # [B, T, vocab_size]
    return output


# -------------------------------
# mBERT (MLM) on XNLI
# -------------------------------

def export_mbert_mlm(args, device: torch.device) -> Dict[str, Any]:
    model_name = 'bert-base-multilingual-cased'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).eval().to(device)
    
    # Get number of transformer layers
    num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else len(model.bert.encoder.layer)
    print(f"[mBERT-MLM] Model: {model_name}, Transformer layers: {num_layers}")

    # Load XNLI with explicit config
    xnli_val = load_dataset('xnli', 'all_languages', split='validation')
    xnli_tst = load_dataset('xnli', 'all_languages', split='test')

    want_langs = set([x.strip() for x in args.mlm_langs.split(',') if x.strip()])

    def lines_from(split):
        out = []
        for ex in split:
            lang = ex.get('language')
            if want_langs and (lang not in want_langs):
                continue
            for key in ('premise', 'hypothesis'):
                s = (ex.get(key) or '').strip()
                if s:
                    out.append(s)
        return out

    lines = lines_from(xnli_val) + lines_from(xnli_tst)
    if not lines:
        # fallback to English-only config
        xnli_val = load_dataset('xnli', 'en', split='validation')
        xnli_tst = load_dataset('xnli', 'en', split='test')
        lines = [(ex['premise'] or '').strip() for ex in xnli_val] + \
                [(ex['hypothesis'] or '').strip() for ex in xnli_val] + \
                [(ex['premise'] or '').strip() for ex in xnli_tst] + \
                [(ex['hypothesis'] or '').strip() for ex in xnli_tst]
        lines = [s for s in lines if s]

    mask_id = tok.mask_token_id
    special_ids = set(tok.all_special_ids)

    outdir = Path(args.outdir)
    logits_path = outdir / f"{args.prefix}_mbert_mlm_logits.bin"
    probs_path  = outdir / f"{args.prefix}_mbert_mlm_probs.bin"

    # Setup timing for softmax operations
    timer = SoftmaxTimer(device)
    original_softmax = wrap_softmax_for_timing(timer)
    print("Softmax timing enabled: tracking F.softmax, nn.functional.softmax, torch.softmax, and tensor.softmax() calls")
    
    written = 0
    input_lengths = []  # Track input token length for each prediction
    # Track statistics for pre-softmax logits (final softmax)
    logit_mins = []  # Min value for each logit vector
    logit_maxs = []  # Max value for each logit vector
    try:
        with open(logits_path, 'wb') as flog, open(probs_path, 'wb') as fprob:
            with torch.inference_mode():
                for text in lines:
                    if written >= args.mlm_max_vectors:
                        break
                    enc = tok(text, return_tensors='pt', truncation=True, max_length=args.mlm_max_seq)
                    inp = enc['input_ids'][0]
                    attn = enc.get('attention_mask', None)

                    ids = inp.tolist()
                    T = len(ids)  # Sequence length
                    cand = [i for i, tid in enumerate(ids) if tid not in special_ids]
                    if not cand:
                        continue
                    idx = cand[len(cand)//2]  # deterministic middle token

                    masked = inp.clone(); masked[idx] = mask_id
                    masked = masked.unsqueeze(0).to(device)
                    if attn is not None: attn = attn.to(device)

                    # Track input length: for MLM, the input is the full sequence length T
                    # (BERT uses bidirectional context, so all tokens are used as input)
                    input_lengths.append(T)

                    # Time the model forward pass (includes softmax in attention and final softmax)
                    timer.start_forward_pass()
                    logits_all = explicit_forward_bert_mlm(model, masked, attention_mask=attn)  # [1, T, V]
                    
                    logits = logits_all[0, idx]  # [V] (≈119,547)
                    
                    # Calculate min and max for this logit vector (before final softmax)
                    logit_mins.append(logits.min().item())
                    logit_maxs.append(logits.max().item())
                    
                    # Time the explicit softmax on final logits (included in forward pass time)
                    timer.start_final_softmax()
                    probs  = F.softmax(logits, dim=-1)
                    timer.end_final_softmax()
                    timer.end_forward_pass()  # End forward pass after final softmax

                    write_record(flog, logits, SOURCE_ID['mbert_mlm'], 0)
                    write_record(fprob, probs,  SOURCE_ID['mbert_mlm'], 0)
                    written += 1
    finally:
        restore_softmax(original_softmax)

    # Get timing statistics
    timing_stats = timer.get_stats()
    
    # Calculate average input token length
    avg_input_length = sum(input_lengths) / len(input_lengths) if input_lengths else 0.0
    min_input_length = min(input_lengths) if input_lengths else 0
    max_input_length = max(input_lengths) if input_lengths else 0
    
    # Calculate statistics for pre-softmax logits (final softmax)
    logit_mins_array = np.array(logit_mins)
    logit_maxs_array = np.array(logit_maxs)
    
    logit_stats = {
        'min': {
            'mean': float(np.mean(logit_mins_array)),
            'std': float(np.std(logit_mins_array)),
            'min': float(np.min(logit_mins_array)),
            'max': float(np.max(logit_mins_array)),
        },
        'max': {
            'mean': float(np.mean(logit_maxs_array)),
            'std': float(np.std(logit_maxs_array)),
            'min': float(np.min(logit_maxs_array)),
            'max': float(np.max(logit_maxs_array)),
        },
        'count': written,
    }
    
    mani = {
        'task': 'mbert_mlm',
        'dataset': 'xnli',
        'model': model_name,
        'vocab_size': int(model.config.vocab_size),
        'count': written,
        'paths': {'logits': str(logits_path), 'probs': str(probs_path)},
        'sha256': {
            'logits': sha256_file(logits_path) if written>0 else '',
            'probs': sha256_file(probs_path) if written>0 else '',
        },
        'record_header': {
            'format': '<IHH',
            'length_bytes': 4,
            'source_bytes': 2,
            'flags_bytes': 2,
            'dtype': 'float32',
        },
        'source_id': SOURCE_ID['mbert_mlm'],
        'mlm_langs': list(want_langs) if want_langs else 'all_languages',
        'max_seq': args.mlm_max_seq,
        'num_transformer_layers': num_layers,
        'input_length_stats': {
            'avg': avg_input_length,
            'min': min_input_length,
            'max': max_input_length,
            'total_predictions': len(input_lengths),
        },
        'logit_stats': logit_stats,
        'timing': timing_stats,
    }
    print(f"[mBERT-MLM] vectors: {written}  (vocab={mani['vocab_size']})")
    print(f"[mBERT-MLM] input token length - avg: {avg_input_length:.2f}, min: {min_input_length}, max: {max_input_length}")
    
    # Print logit statistics
    print("\n==== PRE-SOFTMAX LOGIT STATISTICS (FINAL SOFTMAX): mBERT-MLM ====")
    print(f"Total vectors: {logit_stats['count']}")
    print(f"\nMin value per vector:")
    print(f"  Mean: {logit_stats['min']['mean']:.6f}")
    print(f"  Std:  {logit_stats['min']['std']:.6f}")
    print(f"  Range: [{logit_stats['min']['min']:.6f}, {logit_stats['min']['max']:.6f}]")
    print(f"\nMax value per vector:")
    print(f"  Mean: {logit_stats['max']['mean']:.6f}")
    print(f"  Std:  {logit_stats['max']['std']:.6f}")
    print(f"  Range: [{logit_stats['max']['min']:.6f}, {logit_stats['max']['max']:.6f}]")
    
    # Print timing summary
    _print_timing_summary('mBERT-MLM', timing_stats)
    
    return mani


# -------------------------------
# GPT-2 (CLM) on WikiText-103 test
# -------------------------------

def export_gpt2_clm(args, device: torch.device) -> Dict[str, Any]:
    model_name = 'gpt2'
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)
    
    # Get number of transformer layers
    num_layers = model.config.n_layer if hasattr(model.config, 'n_layer') else len(model.transformer.h)
    print(f"[GPT-2-CLM] Model: {model_name}, Transformer layers: {num_layers}")

    wt = load_dataset('wikitext', 'wikitext-103-v1', split='test')

    outdir = Path(args.outdir)
    logits_path = outdir / f"{args.prefix}_gpt2_clm_logits.bin"
    probs_path  = outdir / f"{args.prefix}_gpt2_clm_probs.bin"

    # Setup timing for softmax operations
    timer = SoftmaxTimer(device)
    original_softmax = wrap_softmax_for_timing(timer)
    print("Softmax timing enabled: tracking F.softmax, nn.functional.softmax, torch.softmax, and tensor.softmax() calls")
    
    written = 0
    input_lengths = []  # Track input token length for each prediction
    # Track statistics for pre-softmax logits (final softmax)
    logit_mins = []  # Min value for each logit vector
    logit_maxs = []  # Max value for each logit vector
    try:
        with open(logits_path, 'wb') as flog, open(probs_path, 'wb') as fprob:
            with torch.inference_mode():
                for ex in wt:
                    if written >= args.clm_max_vectors:
                        break
                    text = (ex['text'] or '').strip()
                    if len(text) < 5:
                        continue
                    enc = tok(text, return_tensors='pt', truncation=True, max_length=args.clm_max_seq)
                    ids = enc['input_ids'].to(device)  # [1, T]
                    T = int(ids.size(1))
                    if T < 2:
                        continue
                    
                    # Time the model forward pass (includes softmax in attention and final softmax)
                    timer.start_forward_pass()
                    logits_all = explicit_forward_gpt2_clm(model, ids)  # [1, T, V]
                    
                    for t in range(0, T-1, args.clm_stride):
                        if written >= args.clm_max_vectors:
                            break
                        logits = logits_all[0, t]  # [V] (=50,257)
                        
                        # Track input length: at position t, we have tokens 0 through t (length = t+1)
                        input_length = t + 1
                        input_lengths.append(input_length)
                        
                        # Calculate min and max for this logit vector (before final softmax)
                        logit_mins.append(logits.min().item())
                        logit_maxs.append(logits.max().item())
                        
                        # Time the explicit softmax on final logits (included in forward pass time)
                        timer.start_final_softmax()
                        probs  = F.softmax(logits, dim=-1)
                        timer.end_final_softmax()
                        
                        write_record(flog, logits, SOURCE_ID['gpt2_clm'], 0)
                        write_record(fprob, probs,  SOURCE_ID['gpt2_clm'], 0)
                        written += 1
                    
                    timer.end_forward_pass()  # End forward pass after all final softmaxes
    finally:
        restore_softmax(original_softmax)

    # Get timing statistics
    timing_stats = timer.get_stats()
    
    # Calculate average input token length
    avg_input_length = sum(input_lengths) / len(input_lengths) if input_lengths else 0.0
    min_input_length = min(input_lengths) if input_lengths else 0
    max_input_length = max(input_lengths) if input_lengths else 0
    
    # Calculate statistics for pre-softmax logits (final softmax)
    logit_mins_array = np.array(logit_mins)
    logit_maxs_array = np.array(logit_maxs)
    
    logit_stats = {
        'min': {
            'mean': float(np.mean(logit_mins_array)),
            'std': float(np.std(logit_mins_array)),
            'min': float(np.min(logit_mins_array)),
            'max': float(np.max(logit_mins_array)),
        },
        'max': {
            'mean': float(np.mean(logit_maxs_array)),
            'std': float(np.std(logit_maxs_array)),
            'min': float(np.min(logit_maxs_array)),
            'max': float(np.max(logit_maxs_array)),
        },
        'count': written,
    }
    
    mani = {
        'task': 'gpt2_clm',
        'dataset': 'wikitext-103-test',
        'model': model_name,
        'vocab_size': int(model.config.vocab_size),
        'count': written,
        'paths': {'logits': str(logits_path), 'probs': str(probs_path)},
        'sha256': {
            'logits': sha256_file(logits_path) if written>0 else '',
            'probs': sha256_file(probs_path) if written>0 else '',
        },
        'record_header': {
            'format': '<IHH',
            'length_bytes': 4,
            'source_bytes': 2,
            'flags_bytes': 2,
            'dtype': 'float32',
        },
        'source_id': SOURCE_ID['gpt2_clm'],
        'stride': args.clm_stride,
        'max_seq': args.clm_max_seq,
        'num_transformer_layers': num_layers,
        'input_length_stats': {
            'avg': avg_input_length,
            'min': min_input_length,
            'max': max_input_length,
            'total_predictions': len(input_lengths),
        },
        'logit_stats': logit_stats,
        'timing': timing_stats,
    }
    print(f"[GPT-2-CLM] vectors: {written}  (vocab={mani['vocab_size']})")
    print(f"[GPT-2-CLM] input token length - avg: {avg_input_length:.2f}, min: {min_input_length}, max: {max_input_length}")
    
    # Print logit statistics
    print("\n==== PRE-SOFTMAX LOGIT STATISTICS (FINAL SOFTMAX): GPT-2-CLM ====")
    print(f"Total vectors: {logit_stats['count']}")
    print(f"\nMin value per vector:")
    print(f"  Mean: {logit_stats['min']['mean']:.6f}")
    print(f"  Std:  {logit_stats['min']['std']:.6f}")
    print(f"  Range: [{logit_stats['min']['min']:.6f}, {logit_stats['min']['max']:.6f}]")
    print(f"\nMax value per vector:")
    print(f"  Mean: {logit_stats['max']['mean']:.6f}")
    print(f"  Std:  {logit_stats['max']['std']:.6f}")
    print(f"  Range: [{logit_stats['max']['min']:.6f}, {logit_stats['max']['max']:.6f}]")
    
    # Print timing summary
    _print_timing_summary('GPT-2-CLM', timing_stats)
    
    return mani


# -------------------------------
# CLI + main
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Export logits & softmax from mBERT (MLM) and GPT-2 (CLM) in the same format as vision exports")
    p.add_argument('--task', required=True, choices=['mbert_mlm', 'gpt2_clm', 'both'])
    p.add_argument('--outdir', required=True)
    p.add_argument('--prefix', default='nlp_export')
    p.add_argument('--cpu', action='store_true')

    # MLM controls
    p.add_argument('--mlm-max-vectors', type=int, default=6000)
    p.add_argument('--mlm-langs', type=str, default='en,de,fr,es,zh')
    p.add_argument('--mlm-max-seq', type=int, default=256)

    # CLM controls
    p.add_argument('--clm-max-vectors', type=int, default=8000)
    p.add_argument('--clm-stride', type=int, default=8)
    p.add_argument('--clm-max-seq', type=int, default=512)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        'version': 1,
        'device': str(device),
        'record_header': {
            'format': '<IHH',
            'length_bytes': 4,
            'source_bytes': 2,
            'flags_bytes': 2,
            'dtype': 'float32',
        }
    }

    if args.task in ('mbert_mlm', 'both'):
        manifest['mbert_mlm'] = export_mbert_mlm(args, device)
    if args.task in ('gpt2_clm', 'both'):
        manifest['gpt2_clm'] = export_gpt2_clm(args, device)

    with open(Path(args.outdir) / f"{args.prefix}_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print("==== DONE ====")


if __name__ == '__main__':
    main()
