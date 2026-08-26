<div align="center">

# 🧠 MiniMind-Next: 全栈轻量 LLM 实验室与 C++ 原生推理引擎

[English](README.md) | [中文](README_zh.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/pytorch-2.6+-orange.svg)](https://pytorch.org/)
[![C++17](https://img.shields.io/badge/c++-17-green.svg)](https://isocpp.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

*专为极简、可掌控与高性能边缘部署而生：涵盖 26M ~ 145M 大模型从零预训练、全阶段微调对齐（SFT/LoRA/DPO/RL/蒸馏）、云端弹性算力流与纯 C++ 零依赖原生推理引擎。*

</div>

---

## 📖 项目简介 (About)

MiniMind-Next 是一个面向研究与工程的轻量级中文大语言模型项目，致力于在极低算力门槛与可控的模型规模（26M ~ 145M）下，提供清晰易读、开箱即用的端到端训练与对齐流水线，以及高性能边缘推理引擎。项目包含：统一的多阶段训练路由（预训练 / SFT / LoRA / DPO / RL / 蒸馏）、一键模型二进制导出工具、以及纯现代 C++ 原生推理引擎（支持 OpenMP 多核加速、GQA、KV Cache 与流式解码）。无论用于学术研究、架构验证还是嵌入式端侧部署，均可提供透明可控的技术基座。

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
├── resource/                   # 权重与训练数据集存放目录 (参考 docs/model_download_zh.md)
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
└── docs/                       # 项目详细文档
    ├── guide_zh.md             # MiniMind-Next 完整使用与训练指南 (中文)
    ├── guide.md                # MiniMind-Next 完整指南 (英文)
    ├── model_download_zh.md    # 模型与数据集下载指南 (中文)
    └── model_download.md       # 模型与数据集下载指南 (英文)
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

> 💡 **资源准备**：预训练权重与数据集请参考 [docs/model_download_zh.md](docs/model_download_zh.md) 快速下载并放置到 `resource/` 目录下即可。

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
# 步骤 1：分别导出 Transformer 核心权重、量化 Embedding 与二进制词表
python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin
python scripts/Tools/export_embedding.py --model_path resource/MiniMind2 --methods fp16 uint4_token int4_group --output_dir models/embedding
python scripts/Tools/export_tokenizer_bin.py --tokenizer_path resource/MiniMind2 --output models/minimind2.vocab.bin

# 步骤 2：使用 CMake Presets 一键编译（可执行文件输出至 bin/）
cmake --preset default
cmake --build --preset default

# 步骤 3：启动极速 C++ 对话终端（可自由指定不同的量化 Embedding 档位）
# 默认使用 FP16 高精度 Embedding：
./bin/minimind_cpp models/minimind2.bin models/embedding/embedding_fp16.embedding models/minimind2.vocab.bin

# 或使用显存极致压缩的 UINT4 per-token 量化 Embedding：
./bin/minimind_cpp models/minimind2.bin models/embedding/embedding_uint4_token.embedding models/minimind2.vocab.bin
```

---

## ⚡ 高性能 Embedding 与 Tokenizer 设计架构

MiniMind-Next 针对端侧嵌入式设备与消费级 CPU 进行了极致解耦与性能优化：

1. **多方案量化 Embedding 矩阵 (`QuantizedEmbedding` / `.embedding`)**：
   - **14 种量化算法支持**：覆盖 Tensor、Token、Group 三种粒度，支持 FP16、NF4、INT4/UINT4、INT8/UINT8 全系列量化。
   - **Zero-Copy MMAP 动态反量化**：通过 Linux `mmap` 进行文件级零拷贝映射 (`LoadMode::DISK`)。首层查表时仅按 Token ID 偏移读取对应的压缩字节并实时反量化为 FP32，极大降低物理内存占用，且无需任何冷启动加载等待。
   - **自包含跨平台**：内建 IEEE 754 float16 解码转换算法，无须依赖特定硬件或外部数学库。

2. **Byte-Level BPE 二进制分词器 (`MMapTokenizer` / `.vocab.bin`)**：
   - **精确 BPE 合并**：基于小顶堆优先队列与字典序二分查找，严格对齐 HuggingFace 官方 Byte-Level BPE 分词逻辑。
   - **零拷贝解码 (`std::string_view`)**：通过紧凑二进制词表映射，单 Token 解码直接返回内存视图，实现零额外内存拷贝。
   - **流式防乱码缓冲**：内置 `decode_stream` 字节边界检测，自动拼接中文汉字等多字节 UTF-8 字符切片，杜绝终端流式打印乱码。

### 📊 实测基准示例：“介绍一下北极”

在标准 CPU 环境（开启 OpenMP 多核并行加速）下实测 MiniMind2 Base 模型（104M 参数，$dim=768$, $layers=16$）：

| 配置档位 | Embedding 文件体积 | Prefill 延迟与吞吐 | Decode 延迟与吞吐 | 实测生成效果 (256 Tokens) |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 原生档位** | **9.38 MB** | 27 tokens / 560.6 ms (**48.16 tok/s**) | 256 tokens / 8583.6 ms (**29.82 tok/s**) | 详细介绍北极的气候、冰盖地理、极地生态圈及北极熊海豹等动物 |
| **UINT4 per-token 量化档位** | **2.37 MB** *(体积压缩 75%)* | 27 tokens / 573.7 ms (**47.06 tok/s**) | 134 tokens / 4213.6 ms (**31.80 tok/s**) | 准确描述北极极地环境、冰冻大陆地理位置及脆弱生态系统的保护意义 |

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

- 📘 [MiniMind-Next 完整使用指南 (中文)](docs/guide_zh.md) | [English Guide](docs/guide.md)
- 📦 [模型与数据集下载指南 (中文)](docs/model_download_zh.md) | [English Guide](docs/model_download.md)

---

## 💖 致谢 (Acknowledgements)

> *Inspired by and evolved from [MiniMind](https://github.com/jingyaogong/minimind), extensively refactored with modular C++ inference and full-stack enhancements.*

感谢原作者 [jingyaogong](https://github.com/jingyaogong) 开创性的 MiniMind 项目为轻量级大模型全流程探索奠定了坚实的基础。本项目在其启发下进行了面向对象工程化重构、算力加速调优、断点续训闭环以及纯 C++ 原生推理引擎的自主研发与拓展。

---

## 📄 License

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。
