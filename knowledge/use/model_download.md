# MiniMind 模型与数据准备指南

> **提示**：当前项目在 `resource/` 目录下已自带一份完整的预训练权重（`resource/MiniMind2-PyTorch/`）、HF 格式模型（`resource/MiniMind2/`）以及常用数据集（`resource/minimind_dataset/`）。
> 如需重新下载、获取更大规模权重或备份，请参考以下方式。

---

## 1. 原生 PyTorch 权重（`.pth` 格式）

用于 `scripts/Deploy/chat_llm.py`、`scripts/Deploy/serve_openai_api.py` 的 `--weight` 原生加载选项。

### 下载方式

```bash
# 方式一：使用 modelscope 命令行（推荐国内用户）
pip install modelscope
modelscope download --model gongjy/MiniMind2-PyTorch --local_dir resource/MiniMind2-PyTorch

# 方式二：使用 git-lfs 克隆
git lfs install
git clone https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch.git resource/MiniMind2-PyTorch
```

### 使用方式

```bash
# 1. 方式 A：直接通过 --save_dir 指向资源目录（最快）
python scripts/Deploy/chat_llm.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512

# 2. 方式 B：将权重软链或复制到 models/ 目录（后续默认直接读取 models/）
ln -s resource/MiniMind2-PyTorch/full_sft_512.pth models/full_sft_512.pth
python scripts/Deploy/chat_llm.py --weight full_sft --hidden_size 512
```

---

## 2. HuggingFace / Transformers 格式模型（用于 `--format hf`）

* **ModelScope**：`https://www.modelscope.cn/models/gongjy/MiniMind2`
* **HuggingFace**：`https://huggingface.co/jingyaogong/MiniMind2`

包含完整的 `config.json`、`tokenizer.json` 及 `model.safetensors`，为标准 `LlamaForCausalLM` 架构。

### 下载与使用

```bash
# 下载至本地（如果 resource/MiniMind2 需重新拉取）
modelscope download --model gongjy/MiniMind2 --local_dir resource/MiniMind2

# 1. Python 端通过 --format hf 直接加载
python scripts/Deploy/chat_llm.py --format hf --load_from resource/MiniMind2

# 2. 导出为 C++ 原生推理 .bin 文件
python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin
./bin/minimind_cpp models/minimind2.bin
```

---

## 3. 权重文件命名规范

原生 `.pth` 权重命名格式：`{save_weight}_{hidden_size}[_moe].pth`

| 文件名示例 | 对应架构参数 | 说明 |
| :--- | :--- | :--- |
| `pretrain_512.pth` | `--weight pretrain --hidden_size 512` | 26M 基础续写预训练权重 |
| `full_sft_512.pth` | `--weight full_sft --hidden_size 512` | 26M 全量指令微调对话权重 |
| `full_sft_768.pth` | `--weight full_sft --hidden_size 768` | 104M Base 规模微调权重 |
| `full_sft_640_moe.pth` | `--weight full_sft --hidden_size 640 --use_moe 1` | 145M 4专家 MoE 结构微调权重 |
| `reason_512.pth` | `--weight reason --hidden_size 512` | 26M 推理思考微调模型 (带 `<think>`) |
| `grpo_512.pth` | `--weight grpo --hidden_size 512` | GRPO 强化学习对齐产物 |

---

## 4. 训练数据集（`resource/minimind_dataset/`）

项目内置的数据集统一存放于 `resource/minimind_dataset/`：

```bash
# 如需拉取最新全量训练数据
modelscope download --dataset gongjy/minimind_dataset --local_dir resource/minimind_dataset
```

| 数据文件名 | 体积与规模 | 适用训练阶段 | 推荐启动命令 |
| :--- | :--- | :--- | :--- |
| `pretrain_t2t_mini.jsonl` | 1.2 GB (127万条) | 预训练快速验证 | `train.py --stage pretrain --data_path resource/minimind_dataset/pretrain_t2t_mini.jsonl` |
| `pretrain_t2t.jsonl` | 10 GB (大语料) | 完整预训练 | `train.py --stage pretrain --data_path resource/minimind_dataset/pretrain_t2t.jsonl` |
| `sft_t2t_mini.jsonl` | 1.6 GB (90万条) | 监督微调 SFT | `train.py --stage full_sft --data_path resource/minimind_dataset/sft_t2t_mini.jsonl` |
| `dpo.jsonl` | ~1.7万条 | DPO 偏好对齐 | `train.py --stage dpo --data_path resource/minimind_dataset/dpo.jsonl` |
| `rlaif.jsonl` | ~1.9万条 | GRPO / PPO / SPO | `train_grpo.py --data_path resource/minimind_dataset/rlaif.jsonl` |
| `lora_identity.jsonl` | 91 条 | LoRA 身份认知微调 | `train.py --stage lora --data_path resource/minimind_dataset/lora_identity.jsonl` |

---

## 5. 云端 / 本地数据同步

如果要在本地与云端 GPU 主机之间同步数据与权重，使用内置的同步工具：

```bash
# 将本地资源推送到云端
python scripts/Tools/sync_data.py push

# 将云端训练出的权重拉回本地
python scripts/Tools/sync_data.py pull --dirs models
```
