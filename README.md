# Minimalist Mathematical Theorem Engine (Qwen2.5-7B + QLoRA)

![Status](https://img.shields.io/badge/status-research--aligned-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Model](https://img.shields.io/badge/model-Qwen2.5--7B--Instruct-blue)
![Training](https://img.shields.io/badge/training-QLoRA%20(4--bit)-orange)
![Platforms](https://img.shields.io/badge/inference-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

一个基于 **Qwen2.5-7B-Instruct** 的数学定理纠错引擎。模型在 7,862 条「错误命题 → 正确命题」对上做 QLoRA 微调，输出严格的 LaTeX 定理表述，**不含任何对话填充**，并通过隐藏的 `<think>` 思维链做内部推理。

> **A 7B math-theorem correction engine fine-tuned with QLoRA on a curated corpus of fallacy-correction pairs. Hidden chain-of-thought, zero-fluff output, runs on a 16 GB Mac or a 6 GB consumer NVIDIA GPU.**

---

## ✨ 核心特性

- **零废话输出**：严格按照「最简协议」工作，只输出修正后的数学命题，不带「我认为」「这个是对的」之类的填充语
- **隐藏的思维链**：通过 `<think>...</think>` 块在内部完成推理与验证，用户只看到最终结果（用 `--show-think` 可查看）
- **抗幻觉训练**：在 ~3,200 条独立的「谬误-修正」配对上训练，避免「跟着用户错下去」
- **跨平台推理**：
  - **Windows / Linux + NVIDIA GPU**：用 `bitsandbytes` 4-bit QLoRA，显存占用 ~5 GB
  - **macOS / Apple Silicon**：用 Apple MLX 4-bit，内存占用 ~5 GB，M2 Pro 实测 37 token/s
- **训练完整可复现**：从原始数据生成、QLoRA 训练、到模型评测全部可一键跑通

---

## 🚀 快速开始

### 一、克隆仓库

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
```

### 二、安装依赖（按你的平台选）

通用依赖：
```bash
pip install -r requirements.txt
```

然后**根据你的平台**装一个：

| 平台 | 安装命令 | 备注 |
|---|---|---|
| **Windows / Linux + NVIDIA GPU** | `pip install -r requirements-windows.txt` | 需要 CUDA 12.1+，至少 6 GB VRAM |
| **macOS Apple Silicon (M1/M2/M3/M4)** | `pip install -r requirements-mac.txt` | 至少 16 GB 统一内存 |
| **AutoDL / 云端训练** | `pip install -r requirements-train.txt` | 24 GB+ NVIDIA GPU |

### 三、获取模型权重

LoRA 适配器（~155 MB）和量化模型（~4 GB）都**不在** Git 仓库里（GitHub 单文件 100 MB 限制）。两种方式获取：

**方式 A（推荐）**：从 HuggingFace Hub 下载预训练好的版本（需作者上传）
```bash
huggingface-cli download <your-hf-id>/qwen-7b-theorem-engine-v2 \
    --local-dir outputs/qwen_7b_minimalist_engine_v2_trained
```

**方式 B**：自己从头训练（云端 RTX 4090 约 2.5 小时，详见下文「训练流程」）。

### 四、运行推理

#### Windows / Linux

```cmd
:: 双击 run_chat_windows.bat
:: 或命令行：
python scripts\inference\chat_windows.py
```

第一次运行会自动从 HuggingFace 下载基座模型（~15 GB）。国内用户建议先：
```bash
set HF_ENDPOINT=https://hf-mirror.com   # Windows
export HF_ENDPOINT=https://hf-mirror.com  # Linux
```

#### macOS

```bash
./run_chat_mac.sh
# 或：
python scripts/inference/chat_mac.py
```

Mac 版需要先把 LoRA 适配器融合到基座模型并转成 MLX 4-bit（一次性，约 15 分钟）：

```bash
# 1. 下基座模型（约 15 GB）
HF_ENDPOINT=https://hf-mirror.com python scripts/training/download_7b.py

# 2. 把 LoRA 适配器融合进基座模型
python scripts/inference/merge_lora.py

# 3. 转换为 MLX 4-bit 量化（约 4 GB）
mlx_lm.convert --hf-path models/qwen_7b_v2_merged \
               --mlx-path models/qwen_7b_v2_mlx_q4 -q --q-bits 4

# 完成。现在可以删 models/Qwen2.5-7B-Instruct/ 和 models/qwen_7b_v2_merged/ 省 30 GB。
python scripts/inference/chat_mac.py
```

### 五、互动示例

```
User: Hey, is it true that every continuous function is d8ff3r%ntiable?
Assistant: Hey, is it true that every differentiable function is continuous?
[4.3s]

User: 1/0 is undefined right?
Assistant: 1/0 is undefined right? Division by zero is undefined in the real numbers.
[3.1s]

User: :think
[CoT visibility: ON]

User: 1/0 is undefined right?
Assistant: <think>
The user asks if 1/0 is undefined. Yes — division by zero is undefined in
standard arithmetic (real numbers, complex numbers). The model should confirm
this without adding fluff.
</think>
1/0 is undefined right? Division by zero is undefined in the real numbers.
```

命令：
- `exit` / `quit` — 退出
- `:think` — 切换是否显示 `<think>` 内部推理（默认隐藏）

---

## 🏗️ 架构概览

### 模型与训练

| 项 | 设置 |
|---|---|
| 基座模型 | `Qwen/Qwen2.5-7B-Instruct` |
| 训练方法 | QLoRA（4-bit NF4 base + bf16 LoRA） |
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | 全部 7 个 linear：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Optimizer | `paged_adamw_32bit` |
| Learning Rate | 1e-4 (cosine schedule, 5% warmup) |
| Effective Batch | 16 (per-device 4 × grad-accum 4) |
| Epochs | 3 |
| Max Sequence Length | 1024 |
| Loss Masking | 仅 assistant 段计算 loss（`-100` 屏蔽 system + user 段） |

### 推理后端

| 平台 | 后端 | 量化 | 内存占用 | 速度 |
|---|---|---|---|---|
| Windows / Linux + NVIDIA | bitsandbytes (`AutoModelForCausalLM`) | NF4 4-bit | ~5 GB VRAM | ~30 t/s (RTX 3060) |
| macOS Apple Silicon | MLX (`mlx_lm`) | 4-bit | ~5 GB unified RAM | ~37 t/s (M2 Pro 16GB) |

---

## 📂 项目结构

```
.
├── README.md                          # 你正在读的文件
├── .gitignore
├── requirements.txt                   # 通用依赖
├── requirements-windows.txt           # Windows / Linux + NVIDIA GPU
├── requirements-mac.txt               # macOS Apple Silicon
├── requirements-train.txt             # AutoDL 云端训练
├── run_chat_windows.bat               # Windows 一键启动
├── run_chat_mac.sh                    # macOS 一键启动
├── autodl_setup.sh                    # AutoDL 一键环境配置 + 训练启动
│
├── data/
│   ├── student_notes_train_v2.json    # 主训练集 (7,862 样本，API 生成)
│   ├── student_notes_train_v2_stats.txt
│   ├── student_notes_train.json       # v1 训练集 (1,979 样本，保留备查)
│   ├── wikipedia_theorems.json        # 数据生成的 Wikipedia 输入
│   └── evaluation/
│       ├── benchmark_150.json         # 主评测集 (150 案例)
│       ├── gemini_deepseek_bench_100.json
│       └── test_cases.json
│
├── scripts/
│   ├── training/
│   │   ├── finetune_7b_qlora.py       # 主训练脚本（本地/单卡）
│   │   ├── finetune_7b_qlora_cloud.py # 24GB+ GPU 云端预设
│   │   ├── callbacks.py               # 进度条 + 样本生成回调
│   │   └── download_7b.py             # 下载 Qwen2.5-7B-Instruct
│   │
│   ├── inference/
│   │   ├── chat_windows.py            # Windows / Linux 推理（4-bit QLoRA）
│   │   ├── chat_mac.py                # macOS 推理（MLX 4-bit）
│   │   └── merge_lora.py              # Mac 用：融合 LoRA → 合并模型
│   │
│   ├── data_processing/
│   │   ├── regenerate_with_teacher.py # 用 DeepSeek-R1 教师生成训练数据
│   │   └── fetch_wikipedia_theorems.py
│   │
│   └── evaluation/
│       ├── evaluate_models.py         # 多模型基准测试
│       ├── data_generator.py
│       └── test_data.jsonl
│
├── outputs/
│   ├── qwen_7b_minimalist_engine_v2_trained/   # 训练好的 LoRA 适配器（git 忽略）
│   └── evaluation/                              # 评测结果（图表 + JSON）
│
└── models/                            # 全部 git 忽略
    └── qwen_7b_v2_mlx_q4/             # MLX 4-bit 量化模型（Mac 推理用）
```

---

## 🧪 训练流程（复现）

### 1. 准备云端机器（AutoDL）

推荐：**RTX 4090 24GB**，单卡，PyTorch 2.3 / CUDA 12.1 镜像。约 ¥2.18/h，整套训练约 ¥5。

### 2. 上传项目

```bash
# 在本地打包
tar --exclude='models/*' --exclude='outputs/*/adapter_model.safetensors' \
    --exclude='__pycache__' --exclude='.DS_Store' \
    -czf upload.tar.gz .

# scp 到 AutoDL
scp -P <port> upload.tar.gz root@<host>:/root/autodl-tmp/

# 在 AutoDL 上解压
cd /root/autodl-tmp && mkdir -p project && tar xzf upload.tar.gz -C project
cd project
```

### 3. 一键训练

```bash
chmod +x autodl_setup.sh
./autodl_setup.sh        # 全自动：装依赖 → 下基座 → smoke test → 正式训练
```

或分阶段：
```bash
./autodl_setup.sh env     # 环境检查
./autodl_setup.sh deps    # 装依赖
./autodl_setup.sh model   # 下基座（5-10 分钟）
./autodl_setup.sh smoke   # 烟雾测试（5-10 分钟）
./autodl_setup.sh train   # 正式训练（1-2 小时）
./autodl_setup.sh fresh   # 不从 checkpoint 续训，从头训
```

### 4. 监控

训练日志会实时打印 `[PROGRESS]` 行，包含 step/total、loss、ETA。也可以开 TensorBoard：
```bash
tensorboard --logdir outputs/qwen_7b_minimalist_engine_v2/runs --host 0.0.0.0 --port 6006
```
然后在 AutoDL 控制台「自定义服务」映射 6006 端口。

### 5. 训练结束自动加载最佳 checkpoint

脚本设置了 `load_best_model_at_end=True` + `metric_for_best_model="eval_loss"`，所以即使后续 epoch 出现过拟合，最终保存的 adapter 也是 eval loss 最低的那个 checkpoint。

### 6. 拉回本地

```bash
scp -P <port> -r root@<host>:/root/autodl-tmp/project/outputs/qwen_7b_minimalist_engine_v2/* \
    ./outputs/qwen_7b_minimalist_engine_v2_trained/
```

> ⚠️ **训练完成后记得在 AutoDL 控制台关机/释放实例**，避免继续计费。

---

## 📊 评测

```bash
python scripts/evaluation/evaluate_models.py
```

评估方法（[evaluate_models.py](scripts/evaluation/evaluate_models.py)）：基于关键词的语义匹配，对每条测试案例验证「必出现关键词集合」全部命中、「禁出现关键词集合」全部未中，公式如下：

$$
\mathcal{S}(y_i) = \left( \prod_{k \in K_{req}} \mathbb{I}[k \in \tilde{y}_i] \right) \cdot
                  \left( \prod_{a \in K_{anti}} (1 - \mathbb{I}[a \in \tilde{y}_i]) \right)
$$

主指标：
- **Theorem Accuracy ($\mathcal{A}_{stmt}$)**：判断陈述是否正确
- **Correction Rate ($\mathcal{A}_{corr}$)**：是否给出正确的修正

评测集见 [data/evaluation/benchmark_150.json](data/evaluation/benchmark_150.json)（150 案例：50 个 Logic Fallacies + 50 个 Missing Conditions + 50 个 Garbled Text）。

---

## 🛠️ 数据生成（可选，进阶）

如果你想用自己的 API key 重新生成训练集：

```bash
# 1. 拉取 Wikipedia 上的数学定理（约 5,000 条）
python scripts/data_processing/fetch_wikipedia_theorems.py

# 2. 用教师模型 (DeepSeek-R1) 生成「学生笔记」训练对
python scripts/data_processing/regenerate_with_teacher.py --target 8000
# 这一步会调用 API，按 token 计费。约 5 小时，可中断续跑。
```

输出会写到 `data/student_notes_train_v2.json`，覆盖现有训练集。

---

## ❓ 常见问题

**Q: 我的 Mac 是 Intel 的，能跑吗？**
A: 不能。MLX 只支持 Apple Silicon。Intel Mac 用户可以用 `llama.cpp` 把模型转成 GGUF 4-bit，但需要自己折腾。

**Q: 我的 GPU 显存只有 4 GB，能跑推理吗？**
A: 太紧。建议至少 6 GB。如果只有 4 GB，可以试 8-bit 而不是 4-bit（修改 `chat_windows.py` 中的 `BitsAndBytesConfig`），但会更慢。

**Q: 训练数据是怎么生成的？**
A: 用 DeepSeek-R1-Distill-Qwen-7B 作为教师模型，给每条 Wikipedia 定理生成 1-2 个常见学生错误版本，并由教师模型生成「正确版本 + 隐藏推理」。详见 [scripts/data_processing/regenerate_with_teacher.py](scripts/data_processing/regenerate_with_teacher.py)。

**Q: 为什么 epoch 2 后 eval loss 反弹？**
A: 这是过拟合。但因为 `load_best_model_at_end=True`，最终 adapter 用的是 eval loss 最低的那个 checkpoint（通常在 epoch 1.5-2 之间）。下次训练可以只跑 1.5-2 epoch。

---

## 📝 License

MIT License — 你可以自由使用、修改、商用。引用本项目请注明出处即可。

## 🙏 致谢

- 基座模型：[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) by Alibaba
- 教师模型：[DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) by DeepSeek
- QLoRA 算法：[Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- LoRA 实现：[HuggingFace PEFT](https://github.com/huggingface/peft)
- Apple Silicon 推理：[MLX](https://github.com/ml-explore/mlx)
- 4-bit 量化：[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
