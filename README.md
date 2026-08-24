<div align="center">

# 🧠 MiniMind-Next: Full-Stack Lightweight LLM Laboratory & Native C++ Inference Engine

[English](README.md) | [中文](README_zh.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/pytorch-2.6+-orange.svg)](https://pytorch.org/)
[![C++17](https://img.shields.io/badge/c++-17-green.svg)](https://isocpp.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

*Engineered for simplicity, complete transparency, and high-performance edge deployment: covering end-to-end pretraining, alignment (SFT/LoRA/DPO/RL/Distillation) for 26M ~ 145M models, elastic cloud training workflows, and a zero-dependency pure C++ native inference engine.*

</div>

---

## About

MiniMind-Next 是一个面向研究与工程的轻量级中文大语言模型项目，致力于在可承受的算力与合理的模型规模下，提供端到端的训练流水线、对齐方法与高性能部署方案。项目包含：统一的训练阶段管理（预训练 / SFT / LoRA / DPO / RL / Distillation）、便捷的模型导出工具、以及一个纯原生 C++ 推理引擎用于低延迟边缘部署与对比实验。文档中提供中英双语使用与部署指南，便于研究复现与工程落地。

MiniMind-Next is a compact Chinese LLM project for research and engineering. It focuses on practical, reproducible workflows for model training, alignment, and deployment on modest hardware. The repository includes a unified staged training pipeline (pretraining, SFT, LoRA, DPO, RL, distillation), model export utilities, and a zero-dependency native C++ inference engine for low-latency edge deployment and performance benchmarking. Bilingual documentation is provided for ease of use and reproducibility.


## 🌟 Key Highlights

- ⚡ **Modular Architecture**: Fully decoupled with object-oriented design across training dispatchers, model definitions, tokenizers, and inference engines.
- 🎯 **Full Training Pipeline**:
  - **Self-Supervised Pretraining**
  - **Supervised Fine-Tuning (Full SFT)**
  - **Parameter-Efficient Tuning (LoRA)**
  - **Direct Preference Optimization (DPO)**
  - **Reinforcement Learning Alignment (GRPO / PPO / SPO)**
  - **Knowledge Distillation & Reasoning SFT**
- 🚀 **Hardware Acceleration**: Deeply integrated with `torch.compile` JIT kernel fusion and mixed precision. Training SFT on an RTX 4090 takes only ~10 minutes.
- ☁️ **Cloud Training Workflow**: Tailored for pay-as-you-go GPU instances with millisecond code hot-sync, automatic Conda environment activation, and `--shutdown` auto power-off.
- 💻 **Pure Native C++ Inference**:
  - Modular, zero-dependency modern C++ inference engine (under [src/](src/)).
  - Native support for **GQA (Grouped Query Attention)**, **KV Cache**, **RoPE (rotate_half)**, and **SwiGLU**.
  - **OpenMP Multi-Core Parallelism**, achieving **13+ tok/s** on CPU with stream buffer handling UTF-8 Chinese characters seamlessly.
  - Built with modern **CMake Presets**.

---

## 📂 Project Structure

```text
.
├── CMakeLists.txt              # C++ root build configuration (outputs to bin/)
├── CMakePresets.json           # Modern CMake presets (default / debug)
├── requirements.txt            # Python dependencies
├── bin/                        # Compiled executable binaries (gitignored)
├── models/                     # Model weights directory (gitignored)
├── checkpoints/                # Resume checkpoints & process snapshots (gitignored)
├── resource/                   # Weights and datasets directory (refer to docs/model_download.md)
│   ├── MiniMind2-PyTorch/      # Native PyTorch weights (.pth)
│   ├── MiniMind2/              # HuggingFace format model (safetensors / config.json)
│   └── minimind_dataset/       # Pretrain/SFT/DPO/RL datasets (.jsonl)
├── src/                        # Pure C++ native inference engine
│   ├── ops/                    # High-performance kernels (RMSNorm, RoPE, MatMul, Softmax)
│   ├── tokenizer/              # Tokenizer & stream decoding (BPE, ChatML template)
│   ├── sampler/                # Sampler (Temperature, Top-P, Greedy)
│   ├── model/                  # Transformer network & KV Cache management
│   └── main.cpp                # CLI interactive chat entry point
├── scripts/
│   ├── Deploy/                 # Python inference & serving (chat_llm, serve_openai_api, chat_openai_api)
│   ├── Trainer/                # Training pipelines (train.py --stage ...) & RL algorithms
│   │   ├── stages/             # Pretrain, SFT, LoRA, DPO, Reason, Distillation implementations
│   │   └── train_common.py     # TrainCtx, resume state, and multi-GPU DDP utilities
│   ├── Tools/                  # Toolings (export_cpp_bin, cloud_train, sync_data, convert_model)
│   ├── Model/                  # Model architecture & Tokenizer
│   └── Dataset/                # Dataset loader (lm_dataset.py)
└── docs/                       # Project documentation
    ├── guide.md                # Full user & training guide (English)
    ├── guide_zh.md             # Full user & training guide (Chinese)
    ├── model_download.md       # Weights & dataset download guide (English)
    └── model_download_zh.md    # Weights & dataset download guide (Chinese)
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/whiteode/MiniMind-Next.git
cd MiniMind-Next

# Install dependencies
pip install -r requirements.txt
```

> 💡 **Resource Preparation**: Please refer to [docs/model_download.md](docs/model_download.md) to download pretrained weights and datasets into `resource/`.

---

### 2. Python Inference & Chat

```bash
# Option 1: Native PyTorch weights terminal chat (with sliding KV cache)
python scripts/Deploy/chat_llm.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512

# Option 2: Load HuggingFace format directory
python scripts/Deploy/chat_llm.py --format hf --load_from resource/MiniMind2

# Option 3: Launch OpenAI-compatible API server (listening on 0.0.0.0:8998)
python scripts/Deploy/serve_openai_api.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512
```

---

### 3. Pure C++ Native Inference (CPU / Zero Dependency)

```bash
# Step 1: Export HuggingFace model into binary format
python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin

# Step 2: Build with CMake Presets (executable generated in bin/)
cmake --preset default
cmake --build --preset default

# Step 3: Launch interactive terminal chat
./bin/minimind_cpp models/minimind2.bin
```

---

## 🏋️ Training from Scratch

All training commands are executed from the **project root directory** via `train.py --stage <stage>`:

### 1. Pretrain
```bash
python scripts/Trainer/train.py --stage pretrain \
  --data_path resource/minimind_dataset/pretrain_t2t_mini.jsonl \
  --batch_size 80 --accumulation_steps 4 --use_compile 1
```

### 2. Full SFT (Supervised Fine-Tuning)
```bash
python scripts/Trainer/train.py --stage full_sft --batch_size 64 --use_compile 1
```

### 3. Advanced Stages
```bash
# LoRA fine-tuning
python scripts/Trainer/train.py --stage lora --batch_size 64 --use_compile 1

# DPO alignment
python scripts/Trainer/train.py --stage dpo --batch_size 8 --use_compile 1

# Knowledge distillation (Student 512 + Teacher 768)
python scripts/Trainer/train.py --stage distillation --batch_size 32 --use_compile 1

# Reinforcement Learning (GRPO / PPO / SPO)
python scripts/Trainer/train_grpo.py
python scripts/Trainer/train_ppo.py
python scripts/Trainer/train_spo.py
```

> 💡 **Resume Training**: Append `--from_resume 1` to any stage to seamlessly resume from `checkpoints/`.

---

## ☁️ Cloud Pay-As-You-Go Training Workflow

```bash
# Sync code -> Activate conda env on remote GPU -> Run SFT -> Auto-shutdown after training
python scripts/Tools/cloud_train.py run minimind python scripts/Trainer/train.py \
  --stage full_sft --batch_size 224 --use_compile 1 --shutdown
```

---

## 📊 Hardware & Batch Size Matrix (Full Param, seq≈340)

| VRAM Specification | 512 (26M Small, Tested) | 768 (104M Base) | 640-MoE (145M) | RTX 4090 Estimated Time |
| :--- | :--- | :--- | :--- | :--- |
| **12GB (3060)** | `batch 80~96` | `batch 32~48` | `batch 32~48` | - |
| **24GB (3090/4090)** | `batch 224` | `batch 96~128` | `batch 128~160` | **Full SFT takes ~10-15 mins** |
| **80GB (A100/H100)** | `batch 768` | `batch 384~512` | `batch 512~640` | **Seconds / Minutes** |

---

## 📖 Documentation

- 📘 [English User Guide](docs/guide.md) | [中文使用指南](docs/guide_zh.md)
- 📦 [English Download Guide](docs/model_download.md) | [中文下载指南](docs/model_download_zh.md)

---

## 💖 Acknowledgements

> *Inspired by and evolved from [MiniMind](https://github.com/jingyaogong/minimind), extensively refactored with modular C++ inference and full-stack enhancements.*

Special thanks to [jingyaogong](https://github.com/jingyaogong) for the inspiring MiniMind project. This project builds upon its foundation with object-oriented refactoring, training acceleration, checkpoint closures, and native C++ inference engine extensions.

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
