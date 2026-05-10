"""
Math Theorem Correction — Interactive Chat (Windows / Linux + NVIDIA GPU)

Loads the Qwen2.5-7B base model in 4-bit (NF4) via bitsandbytes and applies
the trained LoRA adapter on top. Designed for 6-8 GB VRAM consumer GPUs
(GTX 1660, RTX 3060, RTX 4060 and up).

For macOS, use `chat_mac.py` instead — bitsandbytes does NOT support macOS.

CLI:
  python scripts/inference/chat_windows.py
  python scripts/inference/chat_windows.py --adapter outputs/qwen_7b_minimalist_engine_v2_trained
  python scripts/inference/chat_windows.py --show-think    # debug: show CoT
"""

import argparse
import os
import re
import sys
import threading
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

SYSTEM_PROMPT = """# Role
You are a minimalist mathematical theorem engine. Your task is to correct mathematical statements while preserving the user's conversational context.

# Guidelines
1. Logic engine (Hidden CoT): You MUST first wrap your internal reasoning, deductions, and verification of the user's statement inside a <think>...</think> block.
2. Core correction: After thinking, identify the mathematical theorem the user is attempting to state, and replace ONLY the math part with its rigorous, complete form.
3. Garbled text recovery: If the math formulation contains typos or random characters, deduce the intended theorem in your <think> block.
4. Context preservation: If the user includes conversational text (e.g., "I think", "is this right?", "User:"), output that exact conversational text unchanged in its original position outside the <think> block.
5. Zero-fluff AI output: Outside of the <think> block, NEVER add conversational filler or judgments (e.g., "The correct statement is"). Only output the user's text and the corrected theorem.
6. Formatting: Use standard LaTeX formatting for formulas ($...$ or $$...$$).

# Example
Input: "Hey, is it true that every continuous function is d8ff3r%ntiable?"
Output: "<think>
The user asks if continuous implies differentiable. This is mathematically false (Weierstrass function is a counterexample). The correct fundamental theorem relates differentiability to continuity: differentiable implies continuous. I will output the corrected theorem while keeping the user's query wrappers.
</think>
Hey, is it true that every differentiable function is continuous?"
"""


def load_model(base_path: str, adapter_path: str | None):
    print(f"[INFO] Loading tokenizer from {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Configuring 4-bit (NF4) quantization for 8GB VRAM")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[INFO] Loading base model in 4-bit from {base_path} (this takes ~30s)")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[INFO] Base model loaded in {time.time() - t0:.1f}s")

    if adapter_path and os.path.exists(adapter_path):
        print(f"[INFO] Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("[OK] LoRA adapter applied.")
    else:
        if adapter_path:
            print(f"[WARN] Adapter path not found: {adapter_path}. Running base model only.")
        else:
            print("[WARN] No adapter specified. Running base model only.")
        model = base_model

    model.eval()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024**3
        total_gb = total / 1024**3
        print(f"[INFO] GPU memory: {used:.2f} / {total_gb:.2f} GB used")

    return model, tokenizer


def stream_response(model, tokenizer, user_input: str, show_think: bool, max_new_tokens: int):
    """Generate with streaming. Suppress <think>...</think> by default."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        top_p=0.85,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    print("Assistant: ", end="", flush=True)

    buffer = ""
    in_think = False
    think_done = False
    think_started_indicator_printed = False
    full_text = ""

    for chunk in streamer:
        full_text += chunk
        if show_think:
            print(chunk, end="", flush=True)
            continue

        buffer += chunk
        # Detect <think> opening — once seen, suppress until </think> closes.
        if not in_think and not think_done:
            if "<think>" in buffer:
                pre, _, rest = buffer.partition("<think>")
                if pre:
                    print(pre, end="", flush=True)
                buffer = rest
                in_think = True
                if not think_started_indicator_printed:
                    print("[thinking…] ", end="", flush=True)
                    think_started_indicator_printed = True

        if in_think:
            if "</think>" in buffer:
                _, _, post = buffer.partition("</think>")
                buffer = post.lstrip("\n")
                in_think = False
                think_done = True
            else:
                # still inside think — keep buffering, don't print
                continue

        if not in_think and think_done:
            # safe to flush buffer
            if buffer:
                print(buffer, end="", flush=True)
                buffer = ""
        elif not in_think and not think_done:
            # before any <think> — model may not emit one; flush conservatively
            # but only flush up to last newline so we don't accidentally show <thi...
            if "<" not in buffer[-10:]:
                print(buffer, end="", flush=True)
                buffer = ""

    # final flush
    if not show_think:
        if in_think:
            # Model never closed <think>. Try regex fallback on full_text.
            cleaned = re.sub(r"<think>.*?(</think>|$)", "", full_text, flags=re.DOTALL).strip()
            if "<think>" in cleaned:
                cleaned = cleaned.split("</think>")[-1].strip()
            print("\r" + " " * 60 + "\r", end="")
            print(f"Assistant: {cleaned}", end="")
        elif buffer:
            print(buffer, end="", flush=True)

    print()
    thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/Qwen2.5-7B-Instruct",
                        help="Path to base model directory")
    parser.add_argument("--adapter", default="outputs/qwen_7b_minimalist_engine_v2_trained",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--show-think", action="store_true",
                        help="Show the <think> CoT block (for debugging)")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    if not os.path.exists(args.adapter):
        print(f"[WARN] Adapter not found at {args.adapter}. Running base model only.")
        args.adapter = None

    model, tokenizer = load_model(args.base, args.adapter)

    print("\n" + "=" * 60)
    print("Math Assistant - 7B Core Logic Engine (4-bit QLoRA)")
    print("Type 'exit' or 'quit' to leave.")
    print("Type ':think' to toggle CoT visibility.")
    print("=" * 60 + "\n")

    show_think = args.show_think
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            break
        if user_input == ":think":
            show_think = not show_think
            print(f"[CoT visibility: {'ON' if show_think else 'OFF'}]\n")
            continue

        t0 = time.time()
        try:
            stream_response(model, tokenizer, user_input, show_think, args.max_new_tokens)
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            continue
        print(f"[{time.time() - t0:.1f}s]\n")


if __name__ == "__main__":
    main()
