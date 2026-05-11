# Minimalist Mathematical Theorem Engine

![Status](https://img.shields.io/badge/status-research--prototype-brightgreen)
![Base Model](https://img.shields.io/badge/base-Qwen2.5--7B--Instruct-blue)
![Training](https://img.shields.io/badge/training-QLoRA%204--bit-orange)
![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow)
![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

A compact mathematical theorem correction engine fine-tuned from `Qwen/Qwen2.5-7B-Instruct` with QLoRA.

The model is trained to rewrite flawed, incomplete, or garbled mathematical statements into cleaner theorem-style statements while preserving the user's surrounding wording. This GitHub repository contains the code, training data, and inference/training scripts. The trained weights are publicly released on Hugging Face.

## Public Weights

The model weights are not stored directly in this GitHub repository because they are large model artifacts. They are public on Hugging Face:

| Artifact | Hugging Face repository | Use case |
|---|---|---|
| LoRA adapter | [`dots123/qwen-7b-theorem-engine-v2`](https://huggingface.co/dots123/qwen-7b-theorem-engine-v2) | Windows/Linux NVIDIA inference, retraining, merging |
| MLX 4-bit model | [`dots123/qwen-7b-theorem-engine-v2-mlx-q4`](https://huggingface.co/dots123/qwen-7b-theorem-engine-v2-mlx-q4) | Direct macOS Apple Silicon inference |

The MLX model is derived from the same Qwen2.5-7B base model merged with this project's LoRA adapter, then converted to MLX 4-bit format for Apple Silicon.

## Features

- **Theorem correction:** rewrites incorrect, incomplete, or noisy mathematical statements into cleaner theorem-style statements.
- **Context preservation:** keeps surrounding user wording such as "is this right?" or "please check" where possible.
- **QLoRA fine-tuning:** trains a compact LoRA adapter on top of `Qwen/Qwen2.5-7B-Instruct`.
- **Public weights:** LoRA and MLX 4-bit releases are available from Hugging Face.
- **Cross-platform inference:** supports NVIDIA GPU inference through `transformers`/`peft`/`bitsandbytes`, and Apple Silicon inference through MLX.
- **Included training data:** the release dataset is included in `data/student_notes_train_v2.json`.
- **Optional local-AI data generation:** experimental scripts are kept for users who want to extend the dataset with their own local model.

## Installation

Use Python 3.10 or newer.

Install common dependencies:

```bash
pip install -r requirements.txt
```

Install the platform-specific extras you need:

```bash
# Windows / Linux with NVIDIA GPU
pip install -r requirements-windows.txt

# macOS Apple Silicon
pip install -r requirements-mac.txt

# CUDA training environment
pip install -r requirements-train.txt
```

## Quick Start: Windows / Linux NVIDIA

Download the Qwen2.5-7B base model:

```bash
python scripts/training/download_7b.py
```

Download the public LoRA adapter:

```bash
hf download dots123/qwen-7b-theorem-engine-v2 \
  --local-dir outputs/qwen_7b_minimalist_engine_v2_trained
```

Run the chat interface:

```bash
python scripts/inference/chat_windows.py
```

On Windows, you can also double-click:

```text
run_chat_windows.bat
```

## Quick Start: macOS Apple Silicon

The recommended Mac path is to download the public MLX 4-bit model directly:

```bash
pip install -r requirements.txt
pip install -r requirements-mac.txt

hf download dots123/qwen-7b-theorem-engine-v2-mlx-q4 \
  --local-dir models/qwen_7b_v2_mlx_q4

python scripts/inference/chat_mac.py
```

You can also use the launcher:

```bash
./run_chat_mac.sh
```

## Convert the LoRA Adapter to MLX Yourself

If you want to reproduce the MLX release locally, download the base model and LoRA adapter:

```bash
python scripts/training/download_7b.py

hf download dots123/qwen-7b-theorem-engine-v2 \
  --local-dir outputs/qwen_7b_minimalist_engine_v2_trained
```

Merge the LoRA adapter into the base model:

```bash
python scripts/inference/merge_lora.py
```

Convert the merged model to MLX 4-bit:

```bash
mlx_lm.convert \
  --hf-path models/qwen_7b_v2_merged \
  --mlx-path models/qwen_7b_v2_mlx_q4 \
  -q --q-bits 4
```

Then run:

```bash
python scripts/inference/chat_mac.py
```

## Usage

```text
User: Hey, is it true that every continuous function is d8ff3r%ntiable?
Assistant: Hey, is it true that every differentiable function is continuous?

User: Is it true that sqrt(x^2) = x for all real x?
Assistant: Is it true that $\sqrt{x^2} = |x|$ for all real x?
```

Interactive commands:

- `exit`, `quit`, or `:q`: leave the chat.
- `:think`: toggle display of the model's `<think>...</think>` block for debugging.

## Training

The release training data is included at:

```text
data/student_notes_train_v2.json
```

Each record uses this schema:

```json
{
  "instruction": "...system instruction...",
  "input": "...student or user statement...",
  "output": "...target assistant response..."
}
```

Download the base model:

```bash
python scripts/training/download_7b.py
```

Run a short smoke test:

```bash
python scripts/training/finetune_7b_qlora_cloud.py --test
```

Run the full QLoRA training preset:

```bash
python scripts/training/finetune_7b_qlora_cloud.py
```

The default training output directory is:

```text
outputs/qwen_7b_minimalist_engine_v2
```

The inference scripts expect the final adapter at:

```text
outputs/qwen_7b_minimalist_engine_v2_trained
```

## Updating the Public Hugging Face Weights

After retraining, upload the LoRA adapter:

```bash
hf upload dots123/qwen-7b-theorem-engine-v2 \
  outputs/qwen_7b_minimalist_engine_v2_trained .
```

After converting the merged model to MLX 4-bit, upload the Mac model:

```bash
hf upload-large-folder dots123/qwen-7b-theorem-engine-v2-mlx-q4 \
  models/qwen_7b_v2_mlx_q4
```

## Optional Data Generation

The included dataset is already sufficient to reproduce the released training run. The repository also keeps optional local-AI data-generation tooling for users who want to build their own dataset:

```bash
python scripts/data_processing/fetch_wikipedia_theorems.py
python scripts/data_processing/regenerate_with_teacher.py --target 8000
```

This path is experimental and depends on the user's own local model or compatible generation setup. No private API key or credential is included in this repository.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── requirements-windows.txt
├── requirements-mac.txt
├── requirements-train.txt
├── run_chat_windows.bat
├── run_chat_mac.sh
├── autodl_setup.sh
├── data/
│   ├── student_notes_train_v2.json
│   ├── student_notes_train_v2_stats.txt
│   ├── student_notes_train.json
│   └── wikipedia_theorems.json
├── scripts/
│   ├── training/
│   │   ├── download_7b.py
│   │   ├── finetune_7b_qlora.py
│   │   ├── finetune_7b_qlora_cloud.py
│   │   └── callbacks.py
│   ├── inference/
│   │   ├── chat_windows.py
│   │   ├── chat_mac.py
│   │   └── merge_lora.py
│   └── data_processing/
│       ├── fetch_wikipedia_theorems.py
│       └── regenerate_with_teacher.py
├── outputs/
│   └── qwen_7b_minimalist_engine_v2_trained/
└── models/
```

Large model artifacts under `outputs/` and `models/` are ignored by Git and distributed through Hugging Face instead.

## Model Details

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning method | QLoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | `paged_adamw_32bit` |
| Learning rate | `1e-4` |
| Max sequence length | 1024 |
| Training data | `data/student_notes_train_v2.json` |

## Known Limitations

- This is a research prototype, not a formal proof assistant.
- Mathematical outputs should be checked before being used in teaching, writing, or research.
- The model is optimized for theorem-style correction, not general chat.
- The optional data-generation script is kept for experimentation and may need adaptation for a user's local model.

## Acknowledgements

- Base model: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- LoRA/QLoRA tooling: [Hugging Face PEFT](https://github.com/huggingface/peft)
- Transformers runtime: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- NVIDIA 4-bit inference: [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- Apple Silicon inference: [MLX](https://github.com/ml-explore/mlx)

## License

Add a `LICENSE` file before publishing the final GitHub release.
