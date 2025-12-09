#!/bin/bash
# Script to export ViT-B logits and probabilities for CIFAR-10
# Usage: ./run_cifar10.sh

# Configuration - modify these paths as needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-}"  # Change this to your CIFAR-10 data root
OUTDIR="./data/cifar10_vitb"
SPLIT="${SPLIT:-val}"  # or 'train' for training split

# Run the export script
python "${SCRIPT_DIR}/../../data/vit_logits_and_probs_export.py" \
  --dataset cifar10 \
  --split "${SPLIT}" \
  --data_root "${DATA_ROOT}" \
  --hf_model aaraki/vit-base-patch16-224-in21k-finetuned-cifar10 \
  --outdir "${OUTDIR}" \
  --batch_size 128 \
  --num_workers 8 \
  --prefix cifar10_vitb

echo "CIFAR-10 export completed! Results saved to: ${OUTDIR}"
