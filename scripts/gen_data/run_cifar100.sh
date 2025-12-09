#!/bin/bash
# Script to export ViT-B logits and probabilities for CIFAR-100
# Usage: ./run_cifar100.sh

# Configuration - modify these paths as needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-}"  # Change this to your CIFAR-100 data root
OUTDIR="./data/cifar100_vitb"
SPLIT="${SPLIT:-val}"  # or 'train' for training split

# Run the export script
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/../../data/vit_logits_and_probs_export.py" \
  --dataset cifar100 \
  --split "${SPLIT}" \
  --data_root "${DATA_ROOT}" \
  --hf_model pkr7098/cifar100-vit-base-patch16-224-in21k \
  --outdir "${OUTDIR}" \
  --batch_size 128 \
  --num_workers 8 \
  --prefix cifar100_vitb

echo "CIFAR-100 export completed! Results saved to: ${OUTDIR}"
