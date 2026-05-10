"""
QLoRA fine-tuning for Qwen2.5-7B-Instruct -> Math Theorem Correction Engine.

Key fixes vs. the previous version:
  * Loss masking: only the assistant turn contributes to loss; system prompt,
    user input, and pad tokens are masked to -100.
  * Dynamic padding (DataCollatorForSeq2Seq), not max_length padding.
  * LoRA r=16, alpha=32, all 7 linear modules (matches README).
  * gradient_checkpointing for 8GB VRAM.
  * TensorBoard logging at outputs/.../runs.
  * save_total_limit=2 to stop checkpoint disk-bloat.
  * SampleGenerationCallback: prints input/expected/actual every 200 steps so
    you can SEE what the model is learning (or just memorizing).

Default trains on the v2 dataset (data/student_notes_train_v2.json) and writes
the adapter to outputs/qwen_7b_minimalist_engine_v2/. Pass --legacy to fall
back to the original v1 setup.
"""

import argparse
import ctypes
import importlib
import json
import os
import sys
import random

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

# Ensure local imports work whether invoked from project root or scripts/training/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from callbacks import SampleGenerationCallback, ProgressCallback


def check_dependencies():
    required = ["tensorboard", "bitsandbytes", "peft", "accelerate", "datasets", "transformers"]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[FATAL] Missing dependencies: {missing}")
        print(f"[FATAL] Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true",
                   help="Smoke test: 30 samples, 1 epoch, fast")
    p.add_argument("--data", default="data/student_notes_train_v2.json",
                   help="Training data JSON path")
    p.add_argument("--legacy-data", action="store_true",
                   help="Use the original v1 data file at data/student_notes_train.json")
    p.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--output", default="outputs/qwen_7b_minimalist_engine_v2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-every", type=int, default=200,
                   help="Run SampleGenerationCallback every N optimizer steps (0=disable)")
    p.add_argument("--init-from-adapter", default=None,
                   help="Path to an existing LoRA adapter directory to use as initial weights "
                        "(e.g. outputs/qwen_7b_minimalist_engine_v2/checkpoint-200). "
                        "Useful for continuing training after a crash. Optimizer state is NOT "
                        "carried over; the LR schedule restarts. Default: train fresh adapter.")
    p.add_argument("--no-grad-checkpointing", action="store_true",
                   help="Disable gradient checkpointing. ~2x faster forward pass but uses ~2x VRAM. "
                        "Recommended for 24GB+ GPUs (e.g. RTX 4090, A100). Default: enabled (8GB-safe).")
    return p.parse_args()


def load_raw_data(path: str):
    if not os.path.exists(path):
        print(f"[FATAL] Data file not found: {path}")
        print("[HINT] If you haven't generated v2 yet, run:")
        print("       python scripts/data_processing/regenerate_with_teacher.py --target 8000")
        print("       (or pass --legacy-data to use the old data)")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} raw examples from {path}")
    return data


def build_tokenize_fn(tokenizer, max_len: int):
    """
    Returns a function that takes a single example {instruction, input, output}
    and returns {input_ids, labels, attention_mask} with the prompt portion
    masked to -100 in labels (so loss is only on the assistant response).
    """
    def tokenize(example):
        messages_full = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        messages_prompt = messages_full[:-1]

        prompt_text = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Truncate from the right (keep beginning of prompt + as much of response as fits)
        full_ids = full_ids[:max_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        # Safety: lengths must match
        labels = labels[:len(full_ids)]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    return tokenize


def build_eval_examples_for_callback(raw_examples, n=8, seed=42):
    """Pick a few raw examples (system/user/expected) for the sample callback."""
    rng = random.Random(seed)
    picks = rng.sample(raw_examples, min(n, len(raw_examples)))
    return [
        {
            "system": ex["instruction"],
            "user": ex["input"],
            "expected": ex["output"],
        }
        for ex in picks
    ]


def train():
    args = parse_args()
    check_dependencies()

    if args.legacy_data:
        args.data = "data/student_notes_train.json"
        print("[INFO] Legacy mode: using v1 data")

    # Anti-sleep on Windows
    if os.name == "nt":
        print("[INFO] Enabling Windows anti-sleep for the training run")
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    if args.test:
        print("====== TEST MODE ======")
        args.output = args.output + "_test"
        args.epochs = 1

    print(f"[CFG] model={args.model}")
    print(f"[CFG] data={args.data}")
    print(f"[CFG] output={args.output}")
    print(f"[CFG] epochs={args.epochs} lr={args.lr} batch={args.batch} "
          f"grad_accum={args.grad_accum} max_len={args.max_len}")
    print(f"[CFG] lora_r={args.lora_r} lora_alpha={args.lora_alpha}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Data ---
    raw = load_raw_data(args.data)
    if args.test:
        raw = raw[:30]
        print(f"[TEST] Sliced data to {len(raw)} samples")

    # Dedupe by (input, output) just in case
    seen = set()
    deduped = []
    for ex in raw:
        key = (ex.get("input", ""), ex.get("output", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    if len(deduped) != len(raw):
        print(f"[INFO] Deduped {len(raw)} -> {len(deduped)} examples")
    raw = deduped

    # Reserve a few raw examples for the SampleGenerationCallback BEFORE tokenization
    callback_examples = build_eval_examples_for_callback(raw, n=8, seed=args.seed)

    print(f"[INFO] Tokenizing {len(raw)} examples (this may take ~30s)")
    ds = Dataset.from_list(raw)
    tokenize_fn = build_tokenize_fn(tokenizer, max_len=args.max_len)
    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names, num_proc=1)

    # Filter out anything where labels are entirely -100 (would produce NaN loss)
    def has_supervision(ex):
        return any(lbl != -100 for lbl in ex["labels"])
    before = len(tokenized)
    tokenized = tokenized.filter(has_supervision)
    if len(tokenized) != before:
        print(f"[INFO] Filtered {before - len(tokenized)} examples with no supervised tokens")

    split = tokenized.train_test_split(test_size=0.05, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"[INFO] Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    # Quick stat: avg supervised token count
    total_sup = sum(sum(1 for l in ex["labels"] if l != -100) for ex in train_ds.select(range(min(100, len(train_ds)))))
    print(f"[INFO] Avg supervised tokens per (first 100) train ex: "
          f"{total_sup / min(100, len(train_ds)):.1f}")

    # --- Model (4-bit) ---
    print("[INFO] Loading base model in 4-bit (NF4)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    use_grad_ckpt = not args.no_grad_checkpointing
    if not use_grad_ckpt:
        print("[INFO] Gradient checkpointing DISABLED (assumes 24GB+ VRAM)")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_grad_ckpt
    )
    if hasattr(model, "config"):
        # use_cache must be False when grad checkpointing is on; safe to leave off
        # for plain training too (training never benefits from KV cache).
        model.config.use_cache = False

    # --- LoRA ---
    if args.init_from_adapter:
        if not os.path.exists(args.init_from_adapter):
            print(f"[FATAL] --init-from-adapter path not found: {args.init_from_adapter}")
            sys.exit(1)
        adapter_file = os.path.join(args.init_from_adapter, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            print(f"[FATAL] adapter_model.safetensors missing in {args.init_from_adapter}")
            sys.exit(1)
        print(f"[INFO] Loading existing LoRA weights from {args.init_from_adapter}")
        print(f"[INFO] (Optimizer state and LR scheduler restart from step 0; only weights are reused)")
        model = PeftModel.from_pretrained(model, args.init_from_adapter, is_trainable=True)
        # Sanity check
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable == 0:
            print("[FATAL] Loaded adapter has zero trainable parameters. "
                  "Check that the adapter was saved with is_trainable=True.")
            sys.exit(1)
        print(f"[INFO] Adapter loaded as trainable. Trainable params: {trainable:,}")
    else:
        print(f"[INFO] Applying fresh LoRA r={args.lora_r} alpha={args.lora_alpha}, all 7 linear modules")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Trainer ---
    # Compute warmup_steps explicitly (transformers 5.x deprecates warmup_ratio).
    steps_per_epoch = max(1, len(train_ds) // (args.batch * args.grad_accum))
    total_optim_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_optim_steps * 0.05))
    print(f"[INFO] Estimated optimizer steps: {total_optim_steps}, warmup: {warmup_steps}")

    # transformers 5.x prefers TENSORBOARD_LOGGING_DIR env var over logging_dir kwarg
    os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(args.output, "runs")

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,

        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100 if not args.test else 10,
        save_strategy="steps",
        save_steps=200 if not args.test else 50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_32bit",
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else {},

        report_to="tensorboard",
        seed=args.seed,
        data_seed=args.seed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    callbacks = [ProgressCallback()]
    if args.sample_every > 0 and not args.test:
        callbacks.append(SampleGenerationCallback(
            tokenizer=tokenizer,
            eval_examples=callback_examples,
            every_n_steps=args.sample_every,
            num_samples=3,
            max_new_tokens=256,
            seed=args.seed,
        ))
    elif args.test:
        # In test mode, fire it once at step 5 to verify it works
        callbacks.append(SampleGenerationCallback(
            tokenizer=tokenizer,
            eval_examples=callback_examples,
            every_n_steps=5,
            num_samples=1,
            max_new_tokens=128,
            seed=args.seed,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("[INFO] Starting training")
    trainer.train()

    print(f"[INFO] Saving best adapter to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("[OK] Training complete.")

    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    train()
