---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- qwen2.5
- mathematics
- theorem-correction
---

# Qwen 7B Theorem Engine v2 LoRA Adapter

This repository contains the LoRA adapter for a minimalist mathematical theorem correction model fine-tuned from `Qwen/Qwen2.5-7B-Instruct`.

The adapter is trained to correct flawed, incomplete, or garbled mathematical theorem-style statements while preserving the user's surrounding wording.

## Model Contents

This is a PEFT/LoRA adapter, not a full standalone base model. To use it, load it on top of:

```text
Qwen/Qwen2.5-7B-Instruct
```

Expected files include:

- `adapter_config.json`
- `adapter_model.safetensors`
- tokenizer files copied from the training run

## Intended Use

The model is intended for theorem-style mathematical statement correction, such as:

```text
Input:  Is it true that every continuous function is differentiable?
Output: Is it true that every differentiable function is continuous?
```

It is not a proof assistant and should not be treated as an authority for formal verification.

## Quick Start

Install the project dependencies from the companion GitHub repository, download the base model, then download this adapter into the expected local path:

```bash
python scripts/training/download_7b.py

hf download dots123/qwen-7b-theorem-engine-v2 \
  --local-dir outputs/qwen_7b_minimalist_engine_v2_trained

python scripts/inference/chat_windows.py
```

For macOS Apple Silicon, either download the separately converted MLX 4-bit model or merge this adapter into the base model and convert it with `mlx_lm.convert`.

## Training Data

The release training dataset is included in the companion GitHub repository as:

```text
data/student_notes_train_v2.json
```

The data was externally generated and curated for theorem-correction supervision. No private API key, credential, or private generation endpoint is included in the model files.

## Training Configuration

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Method | QLoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Max sequence length | 1024 |

## Limitations

- Outputs should be checked by a human before use.
- The model focuses on mathematical statement correction, not broad tutoring or formal proof checking.
- It may preserve user phrasing even when a more natural standalone theorem statement would be possible.

## Base Model

This adapter depends on [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). Please follow the base model's license and usage terms.
