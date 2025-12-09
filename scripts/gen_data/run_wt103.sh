# ''' WikiText-103 (CLM)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="./data/wt103_gpt2"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/../../data/nlp_logits_and_probs_export.py" \
  --task gpt2_clm \
  --outdir "${OUTDIR}" \
  --prefix wt103_gpt2 \
  --clm-max-vectors 10000 \
  --clm-stride 8 \
  --clm-max-seq 512