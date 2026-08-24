# MiniMind-Next Guide (English Version)

> This guide is intended for users who want to quickly explore or train MiniMind-Next from scratch.
> **Note**: All commands below are executed from the **project root directory**.

```text
Project Structure Overview
├── scripts/
│   ├── Deploy/      Inference/Serving: chat_llm (terminal) / serve_openai_api (API server) / chat_openai_api (client)
│   ├── Trainer/     Training: train.py --stage (pretrain/full_sft/lora/dpo/reason/distillation) + RL (grpo/ppo/spo)
│   ├── Tools/       Utilities: convert_model / export_cpp_bin / cloud_train
│   ├── Model/       Model definitions & tokenizer
│   └── Dataset/     Dataset loader (lm_dataset.py)
├── models/          Model weights directory (training outputs are stored here)
└── resource/        Weights and datasets directory (refer to model_download.md)
```

---

## Chapter 1: Quick Experience (scripts/Deploy)

| Script | Interface | Recommended For |
| --- | --- | --- |
| `chat_llm.py` | Terminal Chat (with sliding KV cache) | Fastest way to test model responses |
| `serve_openai_api.py` | OpenAI-compatible API Server | Integration with 3rd-party UIs / clients |
| `chat_openai_api.py` | Terminal API Client | Chatting via the API server in terminal |

### 0. Preparation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare weights (Refer to docs/model_download.md to download and place into resource/)
ln -s resource/MiniMind2-PyTorch/full_sft_512.pth models/full_sft_512.pth
```

### 1. Terminal Chat: chat_llm.py

```bash
# Chat with pretrained SFT weights (full_sft_512.pth)
python scripts/Deploy/chat_llm.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512

# Raw completion with Pretrain weights
python scripts/Deploy/chat_llm.py --save_dir resource/MiniMind2-PyTorch --weight pretrain --hidden_size 512
```

Load HuggingFace format directly using `--format hf`:

```bash
python scripts/Deploy/chat_llm.py --format hf --load_from resource/MiniMind2
```

### 2. OpenAI-compatible API Server: serve_openai_api.py

```bash
python scripts/Deploy/serve_openai_api.py --save_dir resource/MiniMind2-PyTorch --weight full_sft --hidden_size 512
```

Test with curl:

```bash
curl http://127.0.0.1:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minimind","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Chapter 2: Training MiniMind from Scratch

### 1. Pretraining (`train.py --stage pretrain`)

```bash
# Single GPU training with mini dataset & torch.compile acceleration
python scripts/Trainer/train.py --stage pretrain \
  --data_path resource/minimind_dataset/pretrain_t2t_mini.jsonl \
  --batch_size 80 --accumulation_steps 4 --use_compile 1
```

### 2. Supervised Fine-Tuning SFT (`train.py --stage full_sft`)

```bash
python scripts/Trainer/train.py --stage full_sft --batch_size 64 --use_compile 1
```

### 3. Advanced Training Stages

```bash
# LoRA fine-tuning
python scripts/Trainer/train.py --stage lora --batch_size 64 --use_compile 1

# DPO alignment
python scripts/Trainer/train.py --stage dpo --batch_size 8 --use_compile 1

# Knowledge Distillation
python scripts/Trainer/train.py --stage distillation --batch_size 32 --use_compile 1

# Reinforcement Learning (GRPO / PPO / SPO)
python scripts/Trainer/train_grpo.py
python scripts/Trainer/train_ppo.py
python scripts/Trainer/train_spo.py
```

---

## Chapter 3: Native C++ Inference & Cloud Workflows

### 1. Pure C++ Native Inference (CPU / Zero Dependency)

```bash
# 1. Export HF model to binary format
python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin

# 2. Build via CMake Presets
cmake --preset default
cmake --build --preset default

# 3. Run interactive chat
./bin/minimind_cpp models/minimind2.bin
```

### 2. Cloud Training with Auto-Sync & Auto-Shutdown (`cloud_train.py`)

```bash
# Sync code -> Activate conda env on remote GPU -> Run SFT -> Auto-shutdown after training
python scripts/Tools/cloud_train.py run minimind python scripts/Trainer/train.py \
  --stage full_sft --batch_size 224 --use_compile 1 --shutdown
```
