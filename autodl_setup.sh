#!/usr/bin/env bash
# AutoDL one-shot setup + training launcher.
# Run from project root after uploading files.
#
# Usage on AutoDL:
#   chmod +x autodl_setup.sh
#   ./autodl_setup.sh                 # full pipeline: env -> deps -> model download -> smoke test -> train
#   ./autodl_setup.sh smoke           # only run smoke test (after env+deps+model done)
#   ./autodl_setup.sh train           # only run training (assumes everything else is ready)
#   ./autodl_setup.sh fresh           # train fresh (no --init-from-adapter)

set -euo pipefail

# AutoDL non-interactive SSH shells don't source .bashrc, so prepend miniconda
# explicitly. Harmless if the path doesn't exist.
if [ -d /root/miniconda3/bin ]; then
  export PATH="/root/miniconda3/bin:${PATH}"
fi

STAGE="${1:-all}"
ADAPTER_INIT="outputs/qwen_7b_minimalist_engine_v2/checkpoint-200"
MODEL_DIR="models/Qwen2.5-7B-Instruct"

# Required shard files for the base model
REQUIRED_SHARDS=(
  "$MODEL_DIR/model-00001-of-00004.safetensors"
  "$MODEL_DIR/model-00002-of-00004.safetensors"
  "$MODEL_DIR/model-00003-of-00004.safetensors"
  "$MODEL_DIR/model-00004-of-00004.safetensors"
)

setup_env() {
  echo "=== [1/4] Environment ==="
  export HF_ENDPOINT=https://hf-mirror.com
  echo "HF_ENDPOINT=$HF_ENDPOINT"
  python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'bf16:', torch.cuda.is_bf16_supported())"
}

install_deps() {
  echo "=== [2/4] Installing dependencies ==="
  pip install -q -r requirements.txt
  python -c "import bitsandbytes; print('bitsandbytes OK:', bitsandbytes.__version__)"
}

download_model() {
  echo "=== [3/4] Base model ==="
  local missing=0
  for f in "${REQUIRED_SHARDS[@]}"; do
    if [ ! -f "$f" ]; then
      missing=1
      break
    fi
  done
  if [ "$missing" -eq 0 ]; then
    echo "All 4 shards present, skipping download."
    return
  fi
  echo "Downloading Qwen2.5-7B-Instruct (~15GB) via HF mirror..."
  python scripts/training/download_7b.py
  for f in "${REQUIRED_SHARDS[@]}"; do
    if [ ! -f "$f" ]; then
      echo "[FATAL] Missing after download: $f"
      exit 1
    fi
  done
  echo "Model download complete."
}

run_smoke() {
  echo "=== Smoke test (30 samples, 1 epoch, ~5-10 min) ==="
  python scripts/training/finetune_7b_qlora_cloud.py --test
}

run_train() {
  echo "=== [4/4] Training ==="
  if [ "${1:-resume}" = "fresh" ] || [ ! -d "$ADAPTER_INIT" ]; then
    echo "Training from scratch."
    python scripts/training/finetune_7b_qlora_cloud.py
  else
    echo "Resuming from $ADAPTER_INIT (LoRA weights only; LR/optimizer reset)."
    python scripts/training/finetune_7b_qlora_cloud.py \
      --init-from-adapter "$ADAPTER_INIT"
  fi
}

case "$STAGE" in
  all)
    setup_env
    install_deps
    download_model
    run_smoke
    run_train
    ;;
  env)
    setup_env
    ;;
  deps)
    install_deps
    ;;
  model)
    setup_env
    download_model
    ;;
  smoke)
    setup_env
    run_smoke
    ;;
  train)
    setup_env
    run_train
    ;;
  fresh)
    setup_env
    run_train fresh
    ;;
  *)
    echo "Unknown stage: $STAGE"
    echo "Valid: all | env | deps | model | smoke | train | fresh"
    exit 1
    ;;
esac

echo "=== Done: $STAGE ==="
