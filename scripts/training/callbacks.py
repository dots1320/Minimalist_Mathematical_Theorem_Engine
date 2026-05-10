"""
Custom HuggingFace Trainer callbacks for QLoRA fine-tuning observability.

Callbacks:
  * ProgressCallback: prints a single-line progress update every N steps with
    step/total, percent, epoch, ETA, loss, and lr. Designed for SSH/log-tail
    viewing where tqdm carriage-return rendering is ugly.
  * SampleGenerationCallback: every N optimizer steps, picks a few held-out
    examples, runs `model.generate`, and prints input -> expected -> actual
    to stdout. The single most useful signal for catching "model is just
    memorizing the template" failure modes that loss curves will not reveal.
"""

import random
import time
import torch
from transformers import TrainerCallback


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressCallback(TrainerCallback):
    """Single-line progress logger that survives non-TTY output (SSH, log files)."""

    def __init__(self):
        self._t0 = None
        self._last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()
        self._last_step = 0
        total = state.max_steps if state.max_steps and state.max_steps > 0 else "?"
        print(f"[PROGRESS] Training started — total optimizer steps: {total}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Skip eval-only log events (no train loss key)
        if "loss" not in logs and "train_loss" not in logs:
            return

        step = state.global_step
        total = state.max_steps if state.max_steps else 0
        pct = (step / total * 100.0) if total else 0.0
        epoch = logs.get("epoch", state.epoch or 0.0)
        loss = logs.get("loss", logs.get("train_loss", float("nan")))
        lr = logs.get("learning_rate", float("nan"))

        elapsed = time.time() - (self._t0 or time.time())
        if step > 0 and total:
            steps_per_sec = step / max(elapsed, 1e-6)
            eta = (total - step) / max(steps_per_sec, 1e-6)
            speed = f"{steps_per_sec:.2f}it/s" if steps_per_sec >= 1 else f"{1.0/max(steps_per_sec,1e-6):.1f}s/it"
        else:
            eta = float("nan")
            speed = "?"

        print(
            f"[PROGRESS] step {step}/{total} ({pct:5.1f}%) "
            f"epoch {epoch:5.2f}/{args.num_train_epochs} | "
            f"loss={loss:.4f} lr={lr:.2e} | "
            f"speed={speed} elapsed={_fmt_eta(elapsed)} ETA={_fmt_eta(eta)}",
            flush=True,
        )
        self._last_step = step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            print(f"[PROGRESS] eval @ step {state.global_step}: eval_loss={eval_loss:.4f}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        print(f"[PROGRESS] checkpoint saved @ step {state.global_step} -> {args.output_dir}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - (self._t0 or time.time())
        print(f"[PROGRESS] Training finished. Total wall time: {_fmt_eta(elapsed)}", flush=True)


class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_examples,
        every_n_steps: int = 200,
        num_samples: int = 3,
        max_new_tokens: int = 256,
        seed: int = 42,
    ):
        """
        eval_examples: list of dicts with raw text fields:
            {"system": "...", "user": "...", "expected": "..."}
        """
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples
        self.every_n_steps = every_n_steps
        self.num_samples = min(num_samples, len(eval_examples))
        self.max_new_tokens = max_new_tokens
        self._rng = random.Random(seed)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        if model is None:
            return

        picks = self._rng.sample(self.eval_examples, self.num_samples)
        was_training = model.training
        model.eval()
        try:
            print("\n" + "=" * 70)
            print(f"[SAMPLE GEN @ step {state.global_step}]")
            print("=" * 70)
            for i, ex in enumerate(picks):
                messages = [
                    {"role": "system", "content": ex["system"]},
                    {"role": "user", "content": ex["user"]},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                    )
                gen_only = out[0][inputs.input_ids.shape[1]:]
                actual = self.tokenizer.decode(gen_only, skip_special_tokens=True)
                print(f"\n--- sample {i + 1} ---")
                print(f"USER:     {ex['user'][:160]}")
                print(f"EXPECTED: {ex['expected'][:240]}")
                print(f"ACTUAL:   {actual[:240]}")
            print("=" * 70 + "\n")
        finally:
            if was_training:
                model.train()
