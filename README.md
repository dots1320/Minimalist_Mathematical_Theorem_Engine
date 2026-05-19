# Minimalist Mathematical Theorem Engine

![Status](https://img.shields.io/badge/status-research--prototype-brightgreen)
![Base Model](https://img.shields.io/badge/base-Qwen2.5--7B--Instruct-blue)
![Training](https://img.shields.io/badge/training-QLoRA%204--bit-orange)
![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow)
![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

A compact mathematical statement judgment and correction engine fine-tuned from
`Qwen/Qwen2.5-7B-Instruct`.

Given a mathematical statement, the model classifies it as **CORRECT**,
**FALSE**, **INCOMPLETE**, **GARBLED_BUT_IDENTIFIABLE**, or **UNCLEAR**. It then
returns a clean theorem-style correction, or reports `Cannot identify the
intended theorem.` when the intended theorem is not identifiable.

This repository contains the source code, the public v3 training dataset, and
reproducible training and inference scripts. Large model artifacts are published
separately on Hugging Face and are intentionally not committed to GitHub.

## Release Contents

The current release is **v3**, trained with QLoRA on 8,000 curated examples.

| Artifact | Location | Use case |
|---|---|---|
| LoRA adapter | [`dots123/qwen-7b-theorem-engine-v3`](https://huggingface.co/dots123/qwen-7b-theorem-engine-v3) | Windows / Linux NVIDIA inference, merging, retraining |
| MLX 4-bit model | [`dots123/qwen-7b-theorem-engine-v3-mlx-q4`](https://huggingface.co/dots123/qwen-7b-theorem-engine-v3-mlx-q4) | Direct macOS Apple Silicon inference |
| Public training data | [`data/student_notes_train_v3.json`](data/student_notes_train_v3.json) | Reproducing the fine-tune |

Training summary: 3 epochs, effective batch size 16, 942 optimizer steps, about
1 hour on a single RTX 4090, final eval loss about 0.013.

## Behavior

| Category | Behavior |
|---|---|
| `correct` | Keep a correct theorem concise and clean. |
| `false` | Replace a false statement with a correct theorem or statement. |
| `missing_condition` | Add the missing hypotheses needed for correctness. |
| `garbled_identifiable` | Infer the intended theorem and rewrite it fully. |
| `formal_logic` | Handle formal or domain-specific mathematical statements. |
| `need_check` | Output the fixed fallback when no theorem can be identified. |

The chat scripts hide the model's `<think>...</think>` block by default. Type
`:think` in the interactive chat to show it for debugging. The released chat
scripts normalize the unclear fallback to English: `Cannot identify the intended
theorem.`

## Examples

```text
User: Every continuous function is differentiable.
Assistant: Every differentiable function is continuous.

User: A continuous function on an interval achieves its maximum and minimum.
Assistant: Every continuous function on a closed and bounded interval achieves its maximum and minimum.

User: Pythagorean: a sqrd plus b sqrd is c sqrd in a right triangle.
Assistant: If $a$ and $b$ are the lengths of the legs of a right triangle and $c$ is the length of the hypotenuse, then $a^2 + b^2 = c^2$.

User: Let thing be valid then A iff B maybe with x.
Assistant: Cannot identify the intended theorem.
```

## Installation

Use Python 3.10 or newer.

```bash
pip install -r requirements/requirements.txt

# Then install one platform-specific extra:
pip install -r requirements/requirements-mac.txt        # macOS Apple Silicon, MLX
pip install -r requirements/requirements-windows.txt    # Windows / Linux NVIDIA GPU
pip install -r requirements/requirements-train.txt      # Cloud training on a 24 GB+ NVIDIA GPU
```

Install the Hugging Face CLI if you plan to download or upload artifacts:

```bash
pip install -U huggingface_hub
```

## Quick Start: macOS Apple Silicon

Download the prebuilt MLX 4-bit model and run the chat:

```bash
hf download dots123/qwen-7b-theorem-engine-v3-mlx-q4 \
  --local-dir models/qwen_7b_v3_mlx_q4

python scripts/inference/chat_mac.py
```

You can also use the launcher:

```bash
./run_chat_mac.sh
```

## Quick Start: Windows / Linux NVIDIA

Download the Qwen2.5-7B base model and the v3 LoRA adapter:

```bash
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct

hf download dots123/qwen-7b-theorem-engine-v3 \
  --local-dir outputs/qwen_7b_v3

python scripts/inference/chat_windows.py
```

On Windows, you can also double-click `run_chat_windows.bat`.

## Run a Smoke Test

On macOS with the MLX model downloaded:

```bash
python scripts/inference/smoke_test_v3.py
```

The smoke test runs six canonical prompts and checks for the expected correction
patterns, including the unclear fallback.

## Train v3 Yourself

The full public training dataset is included at
`data/student_notes_train_v3.json`.

Install the training dependencies and download the base model first:

```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-train.txt

HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct
```

Then start the v3 QLoRA training run:

```bash
python scripts/training/finetune_7b_qlora_cloud.py \
  --data data/student_notes_train_v3.json \
  --output outputs/qwen_7b_v3 \
  --epochs 3
```

For a short training sanity check:

```bash
python scripts/training/finetune_7b_qlora_cloud.py --test \
  --data data/student_notes_train_v3.json \
  --output outputs/qwen_7b_v3_smoke
```

The trainer uses 4-bit NF4 QLoRA, LoRA rank 16, paged AdamW, cosine learning
rate scheduling, and an effective batch size of 16 by default.

## Build the MLX Model Yourself

If you trained your own adapter, or want to reproduce the published Mac model,
download the base model and run:

```bash
bash scripts/inference/build_v3_mlx.sh
```

This merges `outputs/qwen_7b_v3/` into the base model, converts the merged model
to MLX 4-bit, and removes the large temporary merged model. The final output is
`models/qwen_7b_v3_mlx_q4/`.

## Dataset Format

Each training record uses this schema:

```json
{
  "category": "missing_condition",
  "instruction": "...system prompt...",
  "input": "...student statement...",
  "output": "<think>\n...\n</think>\nEvery continuous function on a closed and bounded interval achieves its maximum and minimum."
}
```

The `category` field is metadata for analysis. Training uses the
`instruction`, `input`, and `output` fields.

## Repository Layout

```text
.
├── README.md
├── requirements/
│   ├── requirements.txt
│   ├── requirements-mac.txt
│   ├── requirements-windows.txt
│   └── requirements-train.txt
├── run_chat_mac.sh
├── run_chat_windows.bat
├── data/
│   ├── student_notes_train_v3.json
│   ├── student_notes_train_v3_stats.txt
│   └── wikipedia_theorems.json
├── scripts/
│   ├── data_processing/
│   │   ├── fetch_wikipedia_theorems.py
│   │   └── generate_judgment_v3.py
│   ├── inference/
│   │   ├── build_v3_mlx.sh
│   │   ├── chat_mac.py
│   │   ├── chat_windows.py
│   │   ├── merge_lora.py
│   │   └── smoke_test_v3.py
│   └── training/
│       ├── callbacks.py
│       ├── download_7b.py
│       ├── finetune_7b_qlora.py
│       └── finetune_7b_qlora_cloud.py
├── models/      # gitignored local model downloads
└── outputs/     # gitignored local adapters and checkpoints
```

## Model Details

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning method | QLoRA, 4-bit NF4 base with fp16 LoRA weights |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | `paged_adamw_32bit` |
| Learning rate | `1e-4`, cosine schedule |
| Effective batch | 16 |
| Max sequence length | 1024 |
| Epochs | 3 |
| Training records | 8,000 |
| Trainable params | 40,370,176, about 0.53% of the base model |

## Known Limitations

- This is a research prototype, not a formal proof assistant.
- Mathematical outputs should be checked before being used in teaching, writing, or research.
- The model is tuned for compact, single-statement inputs. Long proofs and multi-paragraph documents are outside its main target.
- No model weights are committed to this repository. Download them from Hugging Face or regenerate them locally.

## Acknowledgements

- Base model: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- LoRA / QLoRA tooling: [Hugging Face PEFT](https://github.com/huggingface/peft)
- Transformers runtime: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- NVIDIA 4-bit inference: [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- Apple Silicon inference: [MLX](https://github.com/ml-explore/mlx)

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) for the full
text. The base model `Qwen/Qwen2.5-7B-Instruct` is also Apache-2.0; please
review its license terms before redistributing merged weights.
