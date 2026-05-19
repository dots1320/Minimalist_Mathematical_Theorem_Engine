"""Merge a HF LoRA adapter into base Qwen2.5-7B and save as a single fp16 HF model."""

import argparse
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="outputs/qwen_7b_v3")
    parser.add_argument("--out", default="models/qwen_7b_v3_merged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading base from {args.base} (fp16)")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    print(f"[INFO] Base loaded in {time.time() - t0:.1f}s")

    print(f"[INFO] Loading adapter from {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("[INFO] Merging adapter weights into base...")
    merged = model.merge_and_unload()

    print(f"[INFO] Saving merged model to {args.out}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(args.out)

    print(f"[OK] Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
