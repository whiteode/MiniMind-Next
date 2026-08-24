# Model and Dataset Download Guide (English Version)

> **Note**: All weights and datasets are placed under the `resource/` directory.

---

## 1. Native PyTorch Weights (`.pth` format)

Used with the `--weight` native loading option in `scripts/Deploy/chat_llm.py` and `scripts/Deploy/serve_openai_api.py`.

### Download Methods

```bash
# Method 1: Using ModelScope CLI
pip install modelscope
modelscope download --model gongjy/MiniMind2-PyTorch --local_dir resource/MiniMind2-PyTorch

# Method 2: Using git-lfs
git lfs install
git clone https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch.git resource/MiniMind2-PyTorch
```

### Usage

```bash
# Load directly from the resource directory
python scripts/Deploy/chat_llm.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512
```

---

## 2. HuggingFace / Transformers Format Models (for `--format hf`)

* **ModelScope**: `https://www.modelscope.cn/models/gongjy/MiniMind2`
* **HuggingFace**: `https://huggingface.co/jingyaogong/MiniMind2`

### Download and Usage

```bash
# Download to resource/MiniMind2
modelscope download --model gongjy/MiniMind2 --local_dir resource/MiniMind2

# 1. Python Inference
python scripts/Deploy/chat_llm.py --format hf --load_from resource/MiniMind2

# 2. Export & Run Pure C++ Native Inference
python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin
./bin/minimind_cpp models/minimind2.bin
```

---

## 3. Training Datasets (`resource/minimind_dataset/`)

```bash
modelscope download --dataset gongjy/minimind_dataset --local_dir resource/minimind_dataset
```

| Dataset File | Size & Volume | Stage | Recommended Command |
| :--- | :--- | :--- | :--- |
| `pretrain_t2t_mini.jsonl` | 1.2 GB (1.27M lines) | Pretrain (Fast) | `train.py --stage pretrain --data_path resource/minimind_dataset/pretrain_t2t_mini.jsonl` |
| `pretrain_t2t.jsonl` | 10 GB (Full corpus) | Pretrain (Full) | `train.py --stage pretrain --data_path resource/minimind_dataset/pretrain_t2t.jsonl` |
| `sft_t2t_mini.jsonl` | 1.6 GB (900K lines) | Supervised Fine-Tuning | `train.py --stage full_sft --data_path resource/minimind_dataset/sft_t2t_mini.jsonl` |
| `dpo.jsonl` | ~17K lines | DPO Alignment | `train.py --stage dpo --data_path resource/minimind_dataset/dpo.jsonl` |
| `rlaif.jsonl` | ~19K lines | GRPO / PPO / SPO | `train_grpo.py --data_path resource/minimind_dataset/rlaif.jsonl` |
| `lora_identity.jsonl` | 91 lines | LoRA Identity | `train.py --stage lora --data_path resource/minimind_dataset/lora_identity.jsonl` |
