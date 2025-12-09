#!/bin/bash
# Script to export ViT-B logits and probabilities for ImageNet-1k
# Usage: ./run_imagenet1k.sh

# Configuration - modify these paths as needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGEDIR="${IMAGEDIR:-}"  # Change this to your ImageNet-1k validation directory
OUTDIR="./data/imagenet1k_vitb"
SPLIT="${SPLIT:-val}"

# Run the export script
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/../../data/vit_logits_and_probs_export.py" \
  --dataset imagenet1k \
  --split "${SPLIT}" \
  --imagedir "${IMAGEDIR}" \
  --pretrained \
  --outdir "${OUTDIR}" \
  --batch_size 128 \
  --num_workers 8 \
  --prefix imagenet1k_vitb

echo "ImageNet-1k export completed! Results saved to: ${OUTDIR}"
