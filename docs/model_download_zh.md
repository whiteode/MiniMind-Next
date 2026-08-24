# MiniMind 模型与数据下载指南 (中文版)

> **说明**：权重与数据集统一存放于 `resource/` 目录下。

---

## 1. 原生 PyTorch 权重（`.pth` 格式）

用于 `scripts/Deploy/chat_llm.py`、`scripts/Deploy/serve_openai_api.py` 的 `--weight` 原生加载选项。

### 下载方式

```bash
# 方式一：使用 modelscope 命令行（推荐）
pip install modelscope
modelscope download --model gongjy/MiniMind2-PyTorch --local_dir resource/MiniMind2-PyTorch

# 方式二：使用 git-lfs 克隆
git lfs install
git clone https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch.git resource/MiniMind2-PyTorch
```

### 使用方式

```bash
# 直接指定 --save_dir 指向资源目录
python scripts/Deploy/chat_llm.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512
```

---

## 2. HuggingFace / Transformers 格式模型（用于 `--format hf`）

* **ModelScope**：`https://www.modelscope.cn/models/gongjy/MiniMind2`
* **HuggingFace**：`https://huggingface.co/jingyaogong/MiniMind2`

### 下载与使用

```bash
# 下载至 resource/MiniMind2
modelscope download --model gongjy/MiniMind2 --local_dir resource/MiniMind2

# 1. Python 端加载体验
python scripts/Deploy/chat_llm.py --format hf --load_from resource/MiniMind2

# 2. 导出并运行 C++ 原生推理
python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin
./bin/minimind_cpp models/minimind2.bin
```

---

## 3. 训练数据集（`resource/minimind_dataset/`）

```bash
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
