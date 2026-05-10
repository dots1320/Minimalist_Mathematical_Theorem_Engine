"""Merge HF LoRA adapter into base Qwen2.5-7B and save as a single fp16 HF model."""
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "models/Qwen2.5-7B-Instruct"
ADAPTER = "outputs/qwen_7b_minimalist_engine_v2_trained"
OUT = "models/qwen_7b_v2_merged"

print(f"[INFO] Loading base from {BASE} (fp16)")
t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
)
print(f"[INFO] Base loaded in {time.time() - t0:.1f}s")

print(f"[INFO] Loading adapter from {ADAPTER}")
model = PeftModel.from_pretrained(base, ADAPTER)
print("[INFO] Merging adapter weights into base...")
merged = model.merge_and_unload()

print(f"[INFO] Saving merged model to {OUT}")
Path(OUT).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True)

# Copy tokenizer too
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.save_pretrained(OUT)

print(f"[OK] Merged model saved to {OUT}")
