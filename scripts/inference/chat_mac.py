"""
Math Theorem Correction — Interactive Chat (macOS / Apple Silicon)

Loads a 4-bit MLX-quantized version of the merged (base + LoRA) model.
Designed for M1/M2/M3 Macs with 16+ GB unified memory. Runs at ~30-40 t/s
on M2 Pro 16GB, peak memory ~5 GB.

For Windows or Linux with NVIDIA GPU, use `chat_windows.py` instead —
this script requires Apple's MLX framework (Apple Silicon only).

CLI:
    python scripts/inference/chat_mac.py
    python scripts/inference/chat_mac.py --model models/qwen_7b_v2_mlx_q4
    python scripts/inference/chat_mac.py --show-think

To produce the MLX 4-bit model from the HF LoRA adapter, see:
    scripts/inference/merge_lora.py    (merge HF adapter into HF base)
then:
    mlx_lm.convert --hf-path <merged_hf> --mlx-path <mlx_q4> -q --q-bits 4
"""

import argparse
import os
import re
import sys
import time

from mlx_lm import load, stream_generate

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


def stream_response(model, tokenizer, user_input: str, show_think: bool, max_tokens: int):
    """Stream tokens; suppress <think>...</think> by default."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("Assistant: ", end="", flush=True)

    buffer = ""
    in_think = False
    think_done = False
    think_indicator_printed = False
    full_text = ""

    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        chunk = resp.text
        full_text += chunk

        if show_think:
            print(chunk, end="", flush=True)
            continue

        buffer += chunk

        if not in_think and not think_done:
            if "<think>" in buffer:
                pre, _, rest = buffer.partition("<think>")
                if pre:
                    print(pre, end="", flush=True)
                buffer = rest
                in_think = True
                if not think_indicator_printed:
                    print("[thinking…] ", end="", flush=True)
                    think_indicator_printed = True

        if in_think:
            if "</think>" in buffer:
                _, _, post = buffer.partition("</think>")
                buffer = post.lstrip("\n")
                in_think = False
                think_done = True
            else:
                continue

        if not in_think and think_done:
            if buffer:
                print(buffer, end="", flush=True)
                buffer = ""
        elif not in_think and not think_done:
            if "<" not in buffer[-10:]:
                print(buffer, end="", flush=True)
                buffer = ""

    if not show_think:
        if in_think:
            cleaned = re.sub(r"<think>.*?(</think>|$)", "", full_text, flags=re.DOTALL).strip()
            if "<think>" in cleaned:
                cleaned = cleaned.split("</think>")[-1].strip()
            print("\r" + " " * 60 + "\r", end="")
            print(f"Assistant: {cleaned}", end="")
        elif buffer:
            print(buffer, end="", flush=True)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen_7b_v2_mlx_q4",
                        help="Path to MLX-format model directory (4-bit quantized + LoRA fused)")
    parser.add_argument("--show-think", action="store_true",
                        help="Show the <think> CoT block (debugging)")
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[FATAL] Model directory not found: {args.model}")
        print("[HINT] You need to convert the fused HF model to MLX 4-bit first.")
        print("       See top of this file for the mlx_lm.fuse + mlx_lm.convert commands.")
        sys.exit(1)

    print(f"[INFO] Loading MLX model from {args.model}")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"[INFO] Loaded in {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("Math Assistant - 7B Core Logic Engine (Mac MLX 4-bit)")
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
            stream_response(model, tokenizer, user_input, show_think, args.max_tokens)
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            continue
        print(f"[{time.time() - t0:.1f}s]\n")


if __name__ == "__main__":
    main()
