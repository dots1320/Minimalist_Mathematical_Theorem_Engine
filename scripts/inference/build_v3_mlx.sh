#!/usr/bin/env bash
# Build v3 MLX 4-bit model from base + LoRA adapter.
# Run AFTER downloading models/Qwen2.5-7B-Instruct/ (the fp16 base).
#
# Pipeline:
#   1) merge LoRA v3 adapter into base   -> models/qwen_7b_v3_merged   (~15 GB, temporary)
#   2) convert merged model to MLX 4-bit -> models/qwen_7b_v3_mlx_q4   (~4 GB, keep)
#   3) delete the merged intermediate    -> frees ~15 GB
#
# After it finishes:
#   python scripts/inference/chat_mac.py --model models/qwen_7b_v3_mlx_q4

set -euo pipefail

cd "$(dirname "$0")/../.."

BASE=models/Qwen2.5-7B-Instruct
ADAPTER=outputs/qwen_7b_v3
MERGED=models/qwen_7b_v3_merged
MLX=models/qwen_7b_v3_mlx_q4

if [ ! -f "$BASE/config.json" ] || [ ! -f "$BASE/model.safetensors.index.json" ]; then
  echo "[ERROR] Base model not found at $BASE/"
  echo "[HINT]  Finish the HF download first."
  exit 1
fi
if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "[ERROR] v3 adapter not found at $ADAPTER/"
  exit 1
fi

echo "==> [1/3] Merging LoRA v3 adapter into base (fp16)..."
python scripts/inference/merge_lora.py \
  --base "$BASE" \
  --adapter "$ADAPTER" \
  --out "$MERGED"

echo
echo "==> [2/3] Converting merged HF model to MLX 4-bit..."
rm -rf "$MLX"
python -m mlx_lm convert \
  --hf-path "$MERGED" \
  --mlx-path "$MLX" \
  -q --q-bits 4

echo
echo "==> [3/3] Removing intermediate merged HF model to free ~15 GB..."
rm -rf "$MERGED"

echo
echo "============================================================"
echo "DONE. v3 MLX 4-bit model is at: $MLX"
echo "Run chat:"
echo "  python scripts/inference/chat_mac.py --model $MLX"
echo "============================================================"
