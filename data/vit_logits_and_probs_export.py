#!/usr/bin/env python3
"""
Export pre-softmax logits and post-softmax probabilities for ViT-B models on
CIFAR-10, CIFAR-100, and ImageNet-1k.

- Supports streaming binary output with a simple per-vector header so your C harness
  can read sequentially without numpy.
- Also writes a compact manifest.json with counts and SHA-256 checksums.

USAGE EXAMPLES
--------------
# 1) ImageNet-1k with timm pretrained ViT-B, eval on val split
python vit_logits_and_probs_export.py \
  --dataset imagenet1k --imagedir /path/ILSVRC2012/val \
  --pretrained --outdir ./exports/imagenet1k_vitb

# 2) CIFAR-10 with Hugging Face pretrained model
python vit_logits_and_probs_export.py \
  --dataset cifar10 --data_root /path/cifar10 \
  --hf_model aaraki/vit-base-patch16-224-in21k-finetuned-cifar10 \
  --outdir ./exports/cifar10_vitb

# 3) CIFAR-100 with Hugging Face pretrained model
python vit_logits_and_probs_export.py \
  --dataset cifar100 --data_root /path/cifar100 \
  --hf_model pkr7098/cifar100-vit-base-patch16-224-in21k \
  --outdir ./exports/cifar100_vitb

# 4) CIFAR-10 with your own fine-tuned ViT-B checkpoint
python vit_logits_and_probs_export.py \
  --dataset cifar10 --data_root /path/cifar10 \
  --checkpoint /path/to/vitb_cifar10.pth \
  --outdir ./exports/cifar10_vitb

Notes:
- For CIFAR runs you can use --hf_model for Hugging Face models or --checkpoint for your fine-tuned models.
- Defaults to 224×224 resize and ImageNet mean/std (common for ViT). If your CIFAR
  models were trained with different preprocessing, adjust --img-size/--mean/--std.
"""

import argparse
import json
import os
import struct
import hashlib
import time
import sys
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import transforms

try:
    import timm
except Exception as e:
    raise SystemExit("This script requires 'timm'. Install via: pip install timm")

try:
    from transformers import ViTForImageClassification
except ImportError:
    ViTForImageClassification = None


# -------------------------------
# Binary record format utilities
# -------------------------------
# Header layout (little-endian):
#   uint32 length  -> number of classes
#   uint16 source  -> dataset id
#   uint16 flags   -> dtype & options (0 = fp32)
HEADER_FMT = '<IHH'
HEADER_SIZE = struct.calcsize(HEADER_FMT)

DATASET_ID = {
    'cifar10': 10,
    'cifar100': 11,
    'imagenet1k': 1,
}

DEFAULT_NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet1k': 1000,
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
    """Write one vector (1D tensor) with header to fh.
    Expects vec on CPU, float32.
    """
    assert vec.ndim == 1, f"Expected 1D vector, got shape {tuple(vec.shape)}"
    length = vec.numel()
    header = struct.pack(HEADER_FMT, length, source_id, flags)
    fh.write(header)
    fh.write(vec.numpy().tobytes(order='C'))


# -------------------------------
# Model & data setup
# -------------------------------

def build_model(dataset: str, pretrained: bool, num_classes: int, checkpoint: str = None, 
                hf_model: str = None, device: torch.device = torch.device('cpu')) -> torch.nn.Module:
    """Create ViT-Base (patch16/224) model with appropriate head.
    - ImageNet-1k: you may use timm pretrained weights with --pretrained.
    - CIFAR-10/100: use --hf_model to load from Hugging Face, or --checkpoint for your fine-tuned model.
    
    Recommended HF models:
    - CIFAR-10: aaraki/vit-base-patch16-224-in21k-finetuned-cifar10
    - CIFAR-100: pkr7098/cifar100-vit-base-patch16-224-in21k
    """
    # Use Hugging Face model if specified
    if hf_model:
        if ViTForImageClassification is None:
            raise SystemExit("Hugging Face transformers library is required. Install via: pip install transformers")
        print(f"Loading model from Hugging Face: {hf_model}")
        # Use safetensors to avoid torch.load security vulnerability
        model = ViTForImageClassification.from_pretrained(hf_model, use_safetensors=True)
        model.eval()
        model.to(device)
        # print(model)
        # sys.exit()
        return model
    
    # Otherwise use timm
    model = timm.create_model('vit_base_patch16_224', pretrained=(pretrained and dataset == 'imagenet1k'), num_classes=num_classes)
    model.eval()

    # print the model architecture
    # print(model)

    # sys.exit()

    missing, unexpected = [], []
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if isinstance(ckpt, dict):
            # strip common wrappers like 'module.' from DDP
            ckpt = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in ckpt.items()}
            load = model.load_state_dict(ckpt, strict=False)
            missing = list(load.missing_keys)
            unexpected = list(load.unexpected_keys)
        else:
            raise ValueError("Unsupported checkpoint format: expected dict or {'state_dict': ...}")

    model.to(device)
    if missing or unexpected:
        print("[load_state_dict] missing keys:", missing[:10], '...\n' if len(missing) > 10 else '')
        print("[load_state_dict] unexpected keys:", unexpected[:10], '...\n' if len(unexpected) > 10 else '')
    return model


def build_transform(img_size: int, mean, std, is_imagenet: bool) -> transforms.Compose:
    # For ImageNet eval, common: Resize(256) + CenterCrop(224)
    # For CIFAR, simple Resize to img_size (default 224).
    if is_imagenet:
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def build_dataset(dataset: str, data_root: str, imagedir: str, split: str, transform) -> torch.utils.data.Dataset:
    if dataset == 'cifar10':
        is_train = (split == 'train')
        return tv.datasets.CIFAR10(root=data_root, train=is_train, download=True, transform=transform)
    elif dataset == 'cifar100':
        is_train = (split == 'train')
        return tv.datasets.CIFAR100(root=data_root, train=is_train, download=False, transform=transform)
    elif dataset == 'imagenet1k':
        # Expect ImageFolder and user passes --imagedir (e.g., /path/ILSVRC2012/val)
        if not imagedir:
            raise ValueError("For ImageNet-1k, please specify --imagedir pointing to a folder of images.")
        return tv.datasets.ImageFolder(imagedir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


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


def setup_softmax_hooks(model: torch.nn.Module, timer: SoftmaxTimer) -> List[torch.utils.hooks.RemovableHandle]:
    """Install forward hooks on Attention modules to monitor softmax operations.
    
    This complements the F.softmax wrapper by specifically hooking into
    Attention modules to ensure we capture all attention softmax operations.
    """
    handles = []
    
    def make_attention_hook(name: str, use_cuda: bool):
        """Create a hook for Attention modules to track their softmax usage."""
        def hook(module, input, output):
            # This hook records that we're tracking attention modules
            # The actual softmax timing is done by the F.softmax wrapper
            # This is mainly for verification that we're hooking the right modules
            pass
        return hook
    
    # Register hooks on Attention modules
    attention_count = 0
    for name, module in model.named_modules():
        module_type = type(module).__name__
        # Look for Attention modules (timm uses 'Attention' class name)
        if 'Attention' in module_type or 'attention' in name.lower():
            try:
                hook_fn = make_attention_hook(name, timer.use_cuda_events)
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
                attention_count += 1
            except Exception as e:
                pass  # Skip if hook registration fails
    
    if attention_count > 0:
        print(f"Registered hooks on {attention_count} Attention module(s)")
    
    return handles


class BoundMethodWrapper:
    """Helper to create a bound method wrapper."""
    def __init__(self, wrapper, instance):
        self.wrapper = wrapper
        self.instance = instance
    
    def __call__(self, *args, **kwargs):
        return self.wrapper(self.instance, *args, **kwargs)


class TimedSoftmaxWrapper:
    """Wrapper to track all softmax calls, including those within the model.
    
    This wrapper works as both a function (F.softmax, torch.softmax) and as a method (tensor.softmax).
    """
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
        # Return a bound wrapper that will pass obj as first argument
        return BoundMethodWrapper(self, obj)
        
    def __call__(self, *args, **kwargs):
        # Get caller location for identification
        import inspect
        location = "unknown"
        module_context = ""
        
        # Walk up the call stack to find useful context
        frame = inspect.currentframe().f_back
        stack_depth = 0
        max_depth = 5  # Look up to 5 frames
        
        while frame and stack_depth < max_depth:
            try:
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                func_name = frame.f_code.co_name
                
                # Check if this frame is in an Attention module or related
                if 'attention' in filename.lower() or 'attention' in func_name.lower():
                    module_context = f"[Attention/{func_name}]"
                    break
                
                # Check for timm-specific paths
                if 'timm' in filename.lower():
                    if 'attention' in func_name.lower() or 'attn' in func_name.lower():
                        module_context = f"[timm.Attention/{func_name}]"
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
            # Use CUDA events for accurate GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()  # Ensure previous ops are done
            start_event.record()
            result = self.original_softmax(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed_seconds = elapsed_ms / 1000.0
        else:
            # Use CPU timing
            start_time = time.perf_counter()
            result = self.original_softmax(*args, **kwargs)
            elapsed_seconds = time.perf_counter() - start_time
        
        # Record all softmax calls (will filter later if needed)
        self.timer.record_model_softmax(location, elapsed_seconds)
        
        return result


def wrap_softmax_for_timing(timer: SoftmaxTimer):
    """Wrap all softmax variants to track timing of all softmax calls.
    
    This captures all common softmax call patterns:
    - F.softmax(input, dim) or torch.nn.functional.softmax(input, dim)
    - torch.softmax(input, dim)
    - tensor.softmax(dim)  [Attention modules commonly use this!]
    - nn.functional.softmax(input, dim) [Another common pattern in Attention]
    
    Note: Since F is imported as 'torch.nn.functional', wrapping F.softmax
    automatically wraps torch.nn.functional.softmax (they're the same object).
    
    Returns a tuple of (original_F_softmax, original_torch_softmax, original_tensor_softmax) for restoration.
    """
    original_F_softmax = F.softmax  # This is also torch.nn.functional.softmax
    original_torch_softmax = torch.softmax
    original_tensor_softmax = torch.Tensor.softmax
    
    wrapper_F = TimedSoftmaxWrapper(timer, original_F_softmax, name="F.softmax")
    wrapper_torch = TimedSoftmaxWrapper(timer, original_torch_softmax, name="torch.softmax")
    wrapper_tensor = TimedSoftmaxWrapper(timer, original_tensor_softmax, name="Tensor.softmax")
    
    # Wrap all variants - these all reference the same underlying function objects
    F.softmax = wrapper_F  # Also covers torch.nn.functional.softmax and nn.functional.softmax
    torch.softmax = wrapper_torch
    torch.Tensor.softmax = wrapper_tensor
    
    # Also explicitly wrap torch.nn.functional.softmax for clarity
    # (This is the same object as F.softmax, but being explicit)
    torch.nn.functional.softmax = wrapper_F
    
    return (original_F_softmax, original_torch_softmax, original_tensor_softmax)


def restore_softmax(originals):
    """Restore original softmax functions after timing.
    
    Args:
        originals: Tuple of (original_F_softmax, original_torch_softmax, original_tensor_softmax)
    """
    original_F_softmax, original_torch_softmax, original_tensor_softmax = originals
    F.softmax = original_F_softmax  # Also restores torch.nn.functional.softmax
    torch.nn.functional.softmax = original_F_softmax  # Explicit restoration
    torch.softmax = original_torch_softmax
    torch.Tensor.softmax = original_tensor_softmax


# -------------------------------
# Main export loop
# -------------------------------

def explicit_forward(model, images):
    """Explicit forward pass for VisionTransformer model.
    
    Supports both timm and HuggingFace Transformers ViT architectures.
    Manually calls each component to expose all softmax operations:
    - Patch embedding
    - Positional embeddings and dropout
    - Transformer blocks (each with attention containing softmax)
    - Final normalization
    - Classification head
    """
    # Detect architecture type
    is_hf_model = hasattr(model, 'vit')
    
    if is_hf_model:
        # HuggingFace Transformers architecture
        return _explicit_forward_hf(model, images)
    else:
        # timm architecture
        return _explicit_forward_timm(model, images)


def _explicit_forward_hf(model, images):
    """Explicit forward pass for HuggingFace Transformers ViT model."""
    # 1. Patch embedding: Convert image to patch embeddings
    # HuggingFace uses Conv2d projection
    # Input: [B, 3, H, W] -> Output: [B, num_patches, embed_dim]
    embeddings = model.vit.embeddings
    x = embeddings.patch_embeddings.projection(images)  # [B, embed_dim, H', W']
    B, C, H, W = x.shape
    # Flatten spatial dimensions: [B, embed_dim, H', W'] -> [B, num_patches, embed_dim]
    x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
    
    # 2. Add CLS token (HuggingFace always uses CLS token at the beginning)
    # Check for cls_token attribute (may be stored as parameter or buffer)
    if hasattr(embeddings, 'cls_token'):
        cls_token = embeddings.cls_token
        if cls_token is not None:
            # Expand for batch: [1, 1, embed_dim] -> [B, 1, embed_dim]
            cls_tokens = cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
    
    # 3. Add positional embeddings (applied to all tokens including CLS)
    if hasattr(embeddings, 'position_embeddings') and embeddings.position_embeddings is not None:
        x = x + embeddings.position_embeddings
    
    # 4. Apply dropout
    x = embeddings.dropout(x)
    
    # 5. Transformer encoder layers (each layer contains attention with softmax)
    encoder = model.vit.encoder
    for layer in encoder.layer:
        # HuggingFace ViTLayer structure:
        # - layernorm_before -> attention -> residual
        # - layernorm_after -> mlp -> residual
        
        # Attention branch
        residual = x
        x = layer.layernorm_before(x)  # Pre-norm
        
        # Get attention modules
        attn_module = layer.attention.attention  # ViTSelfAttention
        attn_output = layer.attention.output  # ViTSelfOutput
        
        # Compute Q, K, V separately (HuggingFace uses separate linear layers)
        B, N, C = x.shape
        q = attn_module.query(x)  # [B, N, embed_dim]
        k = attn_module.key(x)    # [B, N, embed_dim]
        v = attn_module.value(x)  # [B, N, embed_dim]
        
        # Determine number of heads and head dimension
        # HuggingFace typically stores this in config or we can infer from dimensions
        if hasattr(attn_module, 'num_attention_heads'):
            num_heads = attn_module.num_attention_heads
        elif hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
        else:
            # Default for ViT-Base: 12 heads
            num_heads = 12
        
        head_dim = C // num_heads
        
        # Reshape Q, K, V: [B, N, embed_dim] -> [B, num_heads, N, head_dim]
        q = q.reshape(B, N, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.reshape(B, N, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        v = v.reshape(B, N, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
        
        # EXPLICIT SOFTMAX: This is what we want to time explicitly!
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N, N]
        
        # Apply attention dropout (if present)
        if hasattr(attn_module, 'dropout'):
            attn_weights = attn_module.dropout(attn_weights)
        
        # Apply attention to values
        x_attn = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        
        # Reshape back: [B, num_heads, N, head_dim] -> [B, N, embed_dim]
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)  # [B, N, embed_dim]
        
        # Apply output projection and dropout
        x_attn = attn_output.dense(x_attn)  # [B, N, embed_dim]
        x_attn = attn_output.dropout(x_attn)
        
        # Add residual connection
        x = x_attn + residual
        
        # MLP branch
        residual = x
        x = layer.layernorm_after(x)  # Post-norm
        
        # Intermediate layer (expand)
        x = layer.intermediate.dense(x)  # [B, N, 3072] for ViT-B
        x = layer.intermediate.intermediate_act_fn(x)  # GELU activation
        
        # Output layer (project back)
        x = layer.output.dense(x)  # [B, N, embed_dim]
        x = layer.output.dropout(x)
        
        # Add residual connection
        x = x + residual
    
    # 6. Final normalization
    x = model.vit.layernorm(x)  # [B, num_patches+1, embed_dim]
    
    # 7. Extract CLS token (HuggingFace always uses CLS token at position 0)
    x = x[:, 0]  # [B, embed_dim]
    
    # 8. Classification head
    output = model.classifier(x)  # [B, num_classes]
    
    return output


def _explicit_forward_timm(model, images):
    """Explicit forward pass for timm ViT model."""
    # 1. Patch embedding: Convert image to patch embeddings
    # Input: [B, 3, H, W] -> Output: [B, num_patches, embed_dim]
    x = model.patch_embed(images)  # [B, num_patches, 768]
    
    # 2. Add CLS token if present (timm ViT typically adds this in patch_embed or separately)
    # Check if model has cls_token attribute
    if hasattr(model, 'cls_token') and model.cls_token is not None:
        # Expand cls_token for batch: [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, 768]
    
    # 3. Add positional embeddings if present
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        x = x + model.pos_embed
    
    # 4. Positional dropout
    x = model.pos_drop(x)
    
    # 5. Patch dropout (typically Identity in inference)
    x = model.patch_drop(x)
    
    # 6. Pre-norm (typically Identity, but call it for completeness)
    x = model.norm_pre(x)
    
    # 7. Transformer blocks (each block contains attention with softmax)
    # This is where softmax operations happen in attention mechanisms
    for block in model.blocks:
        # Explicit block forward pass to expose softmax operations
        residual = x
        
        # Block structure:
        # - norm1 -> attention (with explicit softmax) -> residual
        # - norm2 -> mlp -> residual
        
        # Apply norm1
        x = block.norm1(x)
        
        # Explicit attention forward pass with softmax
        # Get attention module
        attn = block.attn
        
        # Compute QKV: [B, N, embed_dim] -> [B, N, 3*embed_dim]
        B, N, C = x.shape
        qkv = attn.qkv(x)  # [B, N, 2304] for ViT-B (768 * 3)
        
        # Determine number of heads and head dimension
        # timm Attention module typically has num_heads attribute
        if hasattr(attn, 'num_heads'):
            num_heads = attn.num_heads
            head_dim = C // num_heads
        else:
            # Fallback: infer from dimensions (ViT-B typically has 12 heads)
            # qkv output is 3*C, so we can infer num_heads from C
            num_heads = getattr(attn, 'num_heads', 12)  # Default 12 for ViT-B
            head_dim = C // num_heads
        
        # Split into Q, K, V: reshape [B, N, 3*embed_dim] -> [3, B, num_heads, N, head_dim]
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, head_dim]
        
        # Apply q_norm and k_norm (typically Identity)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
        
        # EXPLICIT SOFTMAX: This is what we want to time explicitly!
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N, N]
        
        # Apply attention dropout
        attn_weights = attn.attn_drop(attn_weights)
        
        # Apply attention to values
        x_attn = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        
        # Reshape back: [B, num_heads, N, head_dim] -> [B, N, embed_dim]
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)  # [B, N, embed_dim]
        
        # Project and apply dropout
        x = attn.proj(x_attn)  # [B, N, embed_dim]
        x = attn.proj_drop(x)
        
        # Apply scaling and drop_path (typically Identity in inference)
        x = block.ls1(x)
        x = block.drop_path1(x)
        
        # Add residual connection
        x = x + residual
        
        # MLP branch
        residual = x
        x = block.norm2(x)
        x = block.mlp(x)
        x = block.ls2(x)
        x = block.drop_path2(x)
        x = x + residual
    
    # 8. Final normalization
    x = model.norm(x)  # [B, num_patches+1, 768]
    
    # 9. Extract CLS token or pool (timm typically uses CLS token)
    # If we added CLS token, use it; otherwise pool
    if hasattr(model, 'cls_token') and model.cls_token is not None:
        x = x[:, 0]  # Extract CLS token: [B, 768]
    else:
        # Global average pooling if no CLS token
        x = x.mean(dim=1)  # [B, 768]
    
    # 10. FC norm (typically Identity)
    x = model.fc_norm(x)
    
    # 11. Head dropout
    x = model.head_drop(x)
    
    # 12. Classification head
    output = model.head(x)  # [B, num_classes]
    
    return output

def export_logits_and_probs(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Defaults (ImageNet normalization)
    mean = tuple(map(float, args.mean.split(','))) if args.mean else (0.485, 0.456, 0.406)
    std = tuple(map(float, args.std.split(','))) if args.std else (0.229, 0.224, 0.225)

    num_classes = args.num_classes or DEFAULT_NUM_CLASSES[args.dataset]
    is_imagenet = (args.dataset == 'imagenet1k')

    # Model
    model = build_model(
        dataset=args.dataset,
        pretrained=args.pretrained,
        num_classes=num_classes,
        checkpoint=args.checkpoint,
        hf_model=args.hf_model,
        device=device,
    )

    # Data
    tfm = build_transform(args.img_size, mean, std, is_imagenet)
    dataset = build_dataset(args.dataset, args.data_root, args.imagedir, args.split, tfm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    # Output setup
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logits_path = outdir / f"{args.prefix}_logits.bin"
    probs_path = outdir / f"{args.prefix}_probs.bin"
    manifest_path = outdir / f"{args.prefix}_manifest.json"

    # Streaming write
    sid = DATASET_ID[args.dataset]
    count = 0

    # Track statistics for pre-softmax logits
    logit_mins = []  # Min value for each logit vector
    logit_maxs = []  # Max value for each logit vector

    # Check if model is from Hugging Face (has config attribute)
    is_hf_model = hasattr(model, 'config')
    
    # Setup timing for final softmax operations only
    timer = SoftmaxTimer(device)
    
    # try:
    with open(logits_path, 'wb') as flog, open(probs_path, 'wb') as fprob:
        with torch.inference_mode():
            for i, (images, *rest) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                
                # Model forward pass
                logits = explicit_forward(model, images)  # [B, C]
                
                # Time the explicit softmax on final logits
                timer.start_final_softmax()
                probs = F.softmax(logits, dim=1)
                timer.end_final_softmax()

                logits_cpu = logits.detach().to('cpu', dtype=torch.float32)
                probs_cpu = probs.detach().to('cpu', dtype=torch.float32)

                for b in range(logits_cpu.size(0)):
                    logit_vec = logits_cpu[b]
                    # Calculate min and max for this logit vector
                    logit_mins.append(logit_vec.min().item())
                    logit_maxs.append(logit_vec.max().item())
                    
                    write_record(flog, logit_vec, source_id=sid, flags=0)
                    write_record(fprob, probs_cpu[b], source_id=sid, flags=0)
                    count += 1

                if (i + 1) % max(1, (len(loader)//10)) == 0:
                    print(f"Processed {count} samples...")

    # Get timing statistics for final softmax only
    timing_stats = timer.get_stats()
    
    # Calculate statistics for pre-softmax logits
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
        'count': count,
    }
    
    # Manifest
    mani: Dict[str, Any] = {
        'version': 1,
        'dataset': args.dataset,
        'classes': num_classes,
        'img_size': args.img_size,
        'split': args.split,
        'count': count,
        'paths': {
            'logits': str(logits_path),
            'probs': str(probs_path),
        },
        'sha256': {
            'logits': sha256_file(logits_path),
            'probs': sha256_file(probs_path),
        },
        'record_header': {
            'format': '<IHH',
            'length_bytes': 4,
            'source_bytes': 2,
            'flags_bytes': 2,
            'dtype': 'float32',
        },
        'source_id': sid,
        'pretrained': bool(args.pretrained) or bool(args.hf_model),
        'checkpoint': args.checkpoint or '',
        'hf_model': args.hf_model or '',
        'mean': mean,
        'std': std,
        'final_softmax_timing': {
            'count': timing_stats['final_softmax']['count'],
            'total_seconds': timing_stats['final_softmax']['total_seconds'],
            'mean_seconds': timing_stats['final_softmax']['mean_seconds'],
            'min_seconds': timing_stats['final_softmax']['min_seconds'],
            'max_seconds': timing_stats['final_softmax']['max_seconds'],
        },
        'logit_stats': logit_stats,
    }
    with open(manifest_path, 'w') as f:
        json.dump(mani, f, indent=2)

    print("==== DONE ====")
    print(f"Samples: {count}")
    print(f"Logits:  {logits_path}  (sha256: {mani['sha256']['logits'][:12]}...)")
    print(f"Probs:   {probs_path}   (sha256: {mani['sha256']['probs'][:12]}...)")
    print(f"Manifest:{manifest_path}")
    
    # Print final softmax timing
    final_stats = timing_stats['final_softmax']
    print("\n==== FINAL SOFTMAX TIMING ====")
    print(f"  Count: {final_stats['count']}")
    print(f"  Total time: {final_stats['total_seconds']:.4f}s")
    print(f"  Mean per sample: {final_stats['mean_seconds']*1000:.4f}ms")
    print(f"  Min: {final_stats['min_seconds']*1000:.4f}ms, Max: {final_stats['max_seconds']*1000:.4f}ms")
    
    # Print logit statistics (pre-softmax for final softmax)
    print("\n==== PRE-SOFTMAX LOGIT STATISTICS (FINAL SOFTMAX) ====")
    print(f"Total vectors: {logit_stats['count']}")
    print(f"\nMin value per vector:")
    print(f"  Mean: {logit_stats['min']['mean']:.6f}")
    print(f"  Std:  {logit_stats['min']['std']:.6f}")
    print(f"  Range: [{logit_stats['min']['min']:.6f}, {logit_stats['min']['max']:.6f}]")
    print(f"\nMax value per vector:")
    print(f"  Mean: {logit_stats['max']['mean']:.6f}")
    print(f"  Std:  {logit_stats['max']['std']:.6f}")
    print(f"  Range: [{logit_stats['max']['min']:.6f}, {logit_stats['max']['max']:.6f}]")


# -------------------------------
# CLI
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Export logits and softmax probabilities from ViT-B on CIFAR10/100/ImageNet-1k")
    p.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'imagenet1k'])
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='CIFAR: train/test; ImageNet: typically val')
    p.add_argument('--data_root', default='', help='Root for CIFAR10/100 (torchvision format).')
    p.add_argument('--imagedir', default='', help='Path to ImageFolder for ImageNet-1k split (e.g., /path/ILSVRC2012/val).')

    p.add_argument('--checkpoint', default=None, help='Path to a fine-tuned ViT-B checkpoint (.pth).')
    p.add_argument('--pretrained', action='store_true', help='Use timm pretrained weights (only makes sense for ImageNet-1k).')
    p.add_argument('--hf_model', default=None, help='Load pretrained model from Hugging Face (e.g., aaraki/vit-base-patch16-224-in21k-finetuned-cifar10).')
    p.add_argument('--num_classes', type=int, default=None, help='Override class count if your head differs.')

    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--mean', type=str, default=None, help='Comma-separated RGB means, e.g., 0.485,0.456,0.406')
    p.add_argument('--std', type=str, default=None, help='Comma-separated RGB stds, e.g., 0.229,0.224,0.225')

    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')

    p.add_argument('--outdir', required=True)
    p.add_argument('--prefix', default='vitb_export')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_logits_and_probs(args)
