"""
Math Theorem Correction — Interactive Chat (macOS / Apple Silicon)

Loads a 4-bit MLX-quantized version of the merged (base + LoRA) model.
Designed for M1/M2/M3 Macs with 16+ GB unified memory. Runs at ~30-40 t/s
on M2 Pro 16GB, peak memory ~5 GB.

For Windows or Linux with NVIDIA GPU, use `chat_windows.py` instead —
this script requires Apple's MLX framework (Apple Silicon only).

CLI:
    python scripts/inference/chat_mac.py
    python scripts/inference/chat_mac.py --model models/qwen_7b_v3_mlx_q4
    python scripts/inference/chat_mac.py --show-think

To produce the MLX 4-bit model from the HF LoRA adapter, see:
    scripts/inference/merge_lora.py    (merge HF adapter into HF base)
then:
    mlx_lm.convert --hf-path <merged_hf> --mlx-path <mlx_q4> -q --q-bits 4
"""

import argparse
import os
import re
import select
import sys
import time

from mlx_lm import load, stream_generate

UNCLEAR_ZH = "无法识别其意图定理。"
UNCLEAR_EN = "Cannot identify the intended theorem."


def read_user_input(prompt: str = "User: ", paste_window: float = 0.05) -> str:
    """Read a user line; if more lines are buffered (paste), gather them too.

    When the user pastes a multi-line block, the OS feeds all lines to stdin
    almost instantly. We peek with select() after each line: if more data is
    queued within `paste_window` seconds, keep reading; otherwise return.

    Single-line typing returns immediately because nothing else is queued.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    first = sys.stdin.readline()
    if first == "":
        raise EOFError
    lines = [first.rstrip("\n")]
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], paste_window)
        if not ready:
            break
        nxt = sys.stdin.readline()
        if nxt == "":
            break
        lines.append(nxt.rstrip("\n"))
    return "\n".join(lines).strip()

SYSTEM_PROMPT = """# Role
You are a mathematical statement judgment and correction engine.

# Task
First decide whether the user's mathematical statement is:
CORRECT, FALSE, INCOMPLETE, GARBLED_BUT_IDENTIFIABLE, or UNCLEAR.

# Output rules
1. You MUST first write a concise internal judgment inside a <think>...</think> block.
2. If CORRECT, output the clean mathematical statement only.
3. If FALSE, output the correct theorem or statement.
4. If INCOMPLETE, output the complete rigorous theorem with missing assumptions.
5. If GARBLED_BUT_IDENTIFIABLE, infer the intended theorem and rewrite it fully.
6. If UNCLEAR, output exactly: Cannot identify the intended theorem.
7. Never repeat malformed mathematical text unchanged.
8. Preserve casual wrappers only when they do not damage mathematical rigor.
9. Use standard LaTeX formatting for formulas ($...$ or $$...$$).

# Examples
Input: "Every continuous function is differentiable."
Output: "<think>
Category: FALSE. This reverses the valid implication between differentiability and continuity.
</think>
Every differentiable function is continuous."

Input: "Let thing be valid then A iff B maybe with x."
Output: "<think>
Category: UNCLEAR. The statement does not identify a specific mathematical theorem.
</think>
Cannot identify the intended theorem."
"""


def _to_english_unclear(s: str) -> str:
    return s.replace(UNCLEAR_ZH, UNCLEAR_EN)


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
    # Hold back the last N chars of post-think output so we can rewrite
    # the Chinese UNCLEAR phrase to English even when it streams in chunks.
    HOLD = len(UNCLEAR_ZH)
    pending_tail = ""

    def flush_post_think(chunk_text: str, *, final: bool = False):
        """Append chunk to pending_tail, print everything except the last HOLD chars.
        On final, translate and print the entire remaining tail."""
        nonlocal pending_tail
        pending_tail += chunk_text
        if final:
            print(_to_english_unclear(pending_tail), end="", flush=True)
            pending_tail = ""
            return
        if len(pending_tail) > HOLD:
            emit = pending_tail[:-HOLD]
            pending_tail = pending_tail[-HOLD:]
            print(_to_english_unclear(emit), end="", flush=True)

    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        chunk = resp.text
        full_text += chunk

        if show_think:
            print(_to_english_unclear(chunk), end="", flush=True)
            continue

        buffer += chunk

        if not in_think and not think_done:
            if "<think>" in buffer:
                pre, _, rest = buffer.partition("<think>")
                if pre:
                    flush_post_think(pre)
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
                flush_post_think(buffer)
                buffer = ""
        elif not in_think and not think_done:
            if "<" not in buffer[-10:]:
                flush_post_think(buffer)
                buffer = ""

    if not show_think:
        if in_think:
            cleaned = re.sub(r"<think>.*?(</think>|$)", "", full_text, flags=re.DOTALL).strip()
            if "<think>" in cleaned:
                cleaned = cleaned.split("</think>")[-1].strip()
            print("\r" + " " * 60 + "\r", end="")
            print(f"Assistant: {_to_english_unclear(cleaned)}", end="")
        else:
            flush_post_think("", final=True)
            if buffer:
                print(_to_english_unclear(buffer), end="", flush=True)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen_7b_v3_mlx_q4",
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
            user_input = read_user_input("User: ")
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
