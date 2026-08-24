<div align="center">

# 🧠 MiniMind-Next: 全栈轻量 LLM 实验室与 C++ 原生推理引擎

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/pytorch-2.6+-orange.svg)](https://pytorch.org/)
[![C++17](https://img.shields.io/badge/c++-17-green.svg)](https://isocpp.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

*专为极简、可掌控与高性能边缘部署而生：涵盖 26M ~ 145M 大模型从零预训练、全阶段微调对齐（SFT/LoRA/DPO/RL/蒸馏）、云端弹性算力流与纯 C++ 零依赖原生推理引擎。*

</div>

---

## 🌟 项目亮点

- ⚡ **模块化全栈架构**：代码深度解耦与面向对象重构，训练路由、模型、分词器、推理引擎各司其职，透明无黑盒。
- 🎯 **全链路训练覆盖**：
  - **自监督预训练 (Pretrain)**
  - **全量指令微调 (Full SFT)**
  - **轻量可插拔微调 (LoRA)**
  - **直接偏好优化 (DPO)**
  - **强化学习对齐 (GRPO / PPO / SPO)**
  - **知识蒸馏 (Distillation)** 与 **思维链推理微调 (Reasoning)**
- 🚀 **极速算力优化**：深度集成 `torch.compile` JIT 算子融合与混合精度训练，单卡 RTX 3060 提速 3~4 倍，RTX 4090 上仅需 10 分钟即可跑完指令微调全流程。
- ☁️ **云端按量计费训练流**：专为租赁 GPU（如 4090）打造轻量 `cloud_train.py`，毫秒级代码热同步、Conda 环境自动激活与 `--shutdown` 自动关机防漏扣费。
- 💻 **纯 C++ 原生推理引擎**：
  - 模块化、零第三方依赖的现代 C++ 推理引擎（位于 [src/](src/)）。
  - 原生支持 **GQA（分组查询注意力）**、**KV Cache**、**RoPE 旋转编码 (rotate_half)**、**SwiGLU**。
  - **OpenMP 多核并行加速**，CPU 上达到 **13+ tok/s**，流式输出自动处理 UTF-8 汉字拼接，彻底杜绝乱码。
  - 采用现代 **CMake Presets** 统一构建管理。

---

## 📂 项目结构

```text
.
├── CMakeLists.txt              # C++ 根构建配置（统一输出至 bin/）
├── CMakePresets.json           # 现代 CMake 预设（default / debug）
├── requirements.txt            # Python 依赖清单
├── bin/                        # C++ 编译生成的可执行文件目录 (gitignored)
├── models/                     # 训练产物 / 模型权重存放目录 (gitignored)
├── checkpoints/                # 训练过程现场快照与断点续训存档 (gitignored)
├── resource/                   # 权重与训练数据集存放目录 (参考 model_download.md 下载放置)
│   ├── MiniMind2-PyTorch/      # 原生预训练权重 (.pth)
│   ├── MiniMind2/              # HuggingFace 格式模型 (safetensors / config.json)
│   └── minimind_dataset/       # 预训练/SFT/DPO/RL 训练数据集 (.jsonl)
├── src/                        # 纯 C++ 原生推理引擎源码
│   ├── ops/                    # 高性能算子 (RMSNorm, RoPE, MatMul, Softmax)
│   ├── tokenizer/              # 分词与流式解码引擎 (BPE, ChatML 模板)
│   ├── sampler/                # 采样器 (Temperature, Top-P, Greedy)
│   ├── model/                  # Transformer 神经网络与 KV-Cache 管理
│   └── main.cpp                # 终端交互式对话主程序
├── scripts/
│   ├── Deploy/                 # Python 端推理与服务 (chat_llm, serve_openai_api, chat_openai_api)
│   ├── Trainer/                # 统一训练流水线 (train.py --stage ...) + 强化学习
│   │   ├── stages/             # 预训练、SFT、LoRA、DPO、Reason、Distillation 阶段实现
│   │   └── train_common.py     # 训练上下文 (TrainCtx)、断点恢复与多卡 DDP 工具
│   ├── Tools/                  # 工具集 (export_cpp_bin, cloud_train, sync_data, convert_model)
│   ├── Model/                  # 模型结构与 Tokenizer
│   └── Dataset/                # 数据加载器 (lm_dataset.py)
└── knowledge/
    ├── math/                   # 核心算法推导 (KL 散度证明、DPO/GRPO beta 参数分析)
    └── use/
        ├── guide.md            # 完整使用与训练指南
        └── model_download.md   # 模型与数据集准备指南
```

---

## 🚀 快速上手

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/whiteode/MiniMind-Next.git
cd MiniMind-Next

# 安装 Python 依赖
pip install -r requirements.txt
```

> 💡 **资源准备**：预训练权重与数据集请参考 [知识库 - 模型下载指南](knowledge/use/model_download.md) 快速下载并放置到 `resource/` 目录下即可。

---

### 2. 体验模型对话 (Python 端)

下载权重并放置到 `resource/` 目录后，可直接体验：

```bash
# 方式 1：原生权重终端对话（支持跨轮 KV 缓存）
python scripts/Deploy/chat_llm.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512

# 方式 2：加载 HuggingFace 格式目录
python scripts/Deploy/chat_llm.py --format hf --load_from resource/MiniMind2

# 方式 3：启动 OpenAI 兼容 API 服务 (监听 0.0.0.0:8998)
python scripts/Deploy/serve_openai_api.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512
```

---

### 3. 纯 C++ 原生极速推理 (CPU / 零依赖)

```bash
# 步骤 1：将 HuggingFace 格式模型转为二进制 .bin 格式
python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin

# 步骤 2：使用 CMake Presets 一键编译（可执行文件输出至 bin/）
cmake --preset default
cmake --build --preset default

# 步骤 3：启动极速 C++ 对话终端
./bin/minimind_cpp models/minimind2.bin
```

---

## 🏋️ 模型从零训练

所有训练均在**项目根目录**下执行，统一采用 `train.py --stage <阶段>`：

### 1. 预训练 (Pretrain)
```bash
# 单卡极致加速（3060 12GB 约 40~50min/epoch）
python scripts/Trainer/train.py --stage pretrain \
  --data_path resource/minimind_dataset/pretrain_t2t_mini.jsonl \
  --batch_size 80 --accumulation_steps 4 --use_compile 1
```

### 2. 全量指令微调 (Full SFT)
```bash
# 基于 pretrain 权重做指令微调
python scripts/Trainer/train.py --stage full_sft --batch_size 64 --use_compile 1
```

### 3. 进阶对齐训练
```bash
# LoRA 身份微调
python scripts/Trainer/train.py --stage lora --batch_size 64 --use_compile 1

# DPO 人类偏好对齐
python scripts/Trainer/train.py --stage dpo --batch_size 8 --use_compile 1

# 知识蒸馏 (学生 512 + 教师 768)
python scripts/Trainer/train.py --stage distillation --batch_size 32 --use_compile 1

# 强化学习 (GRPO / PPO / SPO)
python scripts/Trainer/train_grpo.py
python scripts/Trainer/train_ppo.py
python scripts/Trainer/train_spo.py
```

> 💡 **断点续训**：任意阶段均可添加 `--from_resume 1`，自动从 `checkpoints/` 读取进度无缝续训。

---

## ☁️ 云端按量计费训练工作流

针对租赁按量计费的云 GPU（如 RTX 4090），项目内置了轻量化一键同步与远程执行工具 `cloud_train.py`：

```bash
# 格式：python scripts/Tools/cloud_train.py run <conda环境名> <命令...>

# 示例：本地一条命令同步代码 -> 云端激活 minimind 环境 -> 开启 4090 极速 SFT -> 训练完自动关机
python scripts/Tools/cloud_train.py run minimind python scripts/Trainer/train.py \
  --stage full_sft --batch_size 224 --use_compile 1 --shutdown
```

*详细配置请参考 [scripts/Tools/cloud_config.example.py](scripts/Tools/cloud_config.example.py)。*

---

## 📊 硬件规格与 Batch 对照表 (全参训练，seq≈340)

| 显存规格 | 512 (26M Small, 实测) | 768 (104M Base) | 640-MoE (145M) | 4090 预估耗时 |
| :--- | :--- | :--- | :--- | :--- |
| **12GB (3060)** | `batch 80~96` | `batch 32~48` | `batch 32~48` | - |
| **24GB (3090/4090)** | `batch 224` | `batch 96~128` | `batch 128~160` | **Full SFT 仅需 ~10-15 分钟** |
| **80GB (A100/H100)** | `batch 768` | `batch 384~512` | `batch 512~640` | **秒级/分钟级收敛** |

---

## 📖 文档导航

- 📘 [MiniMind-Next 完整使用指南](knowledge/use/guide.md)：从零体验、各阶段训练冒烟与参数调优全解析。
- 📦 [模型与数据集下载指南](knowledge/use/model_download.md)：权重下载、数据格式与文件组织。
- 📐 [KL 散度无偏估计推导](knowledge/math/kl散度无偏估计证明.md) & [DPO/GRPO Beta 参数分析](knowledge/math/dpo_grpo_beta.md)。

---

## 💖 致谢 (Acknowledgements)

> *Inspired by and evolved from [MiniMind](https://github.com/jingyaogong/minimind), extensively refactored with modular C++ inference and full-stack enhancements.*

感谢原作者 [jingyaogong](https://github.com/jingyaogong) 开创性的 MiniMind 项目为轻量级大模型全流程探索奠定了坚实的基础。本项目在其启发下进行了面向对象工程化重构、算力加速调优、断点续训闭环以及纯 C++ 原生推理引擎的自主研发与拓展。

---

## 📄 License

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。
