# MiniMind — Agent Quick-Start

## What This Repo Is

Educational LLM project training a 25-104M parameter model from scratch in raw PyTorch. Full pipeline: tokenizer → pretrain → SFT → LoRA → DPO → RL (PPO/GRPO/SPO) → reasoning → distillation. Chinese-authored, Chinese comments throughout.

## Project Structure

```
trainer/          All training scripts (standalone .py, not installed as a package)
model/            Model architecture (model_minimind.py) + LoRA (model_lora.py) + tokenizer files
dataset/          Dataset classes (PretrainDataset, SFTDataset, DPODataset, RLAIFDataset)
dataset/*.jsonl   Training data files (gitignored)
out/              Pre-trained weight checkpoints (.pth) — committed via git LFS
scripts/          Utility scripts (convert, serve, chat)
minimind_dataset/ Git submodule with actual .jsonl data
plan/             Per-script study plans + learning_checklist.md
eval_llm.py       Inference / chat entry point (851 lines)
```

## Running Training Scripts

All scripts are run from `trainer/` directory. They use `sys.path.append('..')` for imports — not installed as a package.

```bash
cd trainer/

# Pretrain
python train_pretrain.py --hidden_size 512 --epochs 5

# SFT (requires pretrain checkpoint)
python train_full_sft.py --hidden_size 512 --from_weight pretrain

# All other stages (require previous stage's checkpoint)
python train_lora.py --from_weight full_sft
python train_dpo.py --from_weight full_sft
python train_reason.py --from_weight full_sft
python train_grpo.py --from_weight full_sft
python train_ppo.py --from_weight full_sft
python train_spo.py --from_weight full_sft
python train_distillation.py --from_weight full_sft
python train_tokenizer.py   # No GPU needed

# Multi-GPU
torchrun --nproc_per_node=N train_pretrain.py ...

# Inference
cd .. && python eval_llm.py --weight full_sft --hidden_size 512
```

## Weight File Convention

Weights are saved to `out/` with naming: `{stage}_{hidden_size}[_moe].pth`

Examples: `pretrain_512.pth`, `full_sft_768.pth`, `grpo_640_moe.pth`

The `--from_weight` arg selects which checkpoint stage to load. The config is inferred from filename, not a separate config arg.

## Key Architecture Details

- LLaMA-style decoder-only Transformer with pre-RMSNorm, SwiGLU, GQA, RoPE
- Config sizes: Small=512/8L (26M), Base=768/16L (104M), MoE=640/8L (145M)
- Vocab size: 6400, max context: 32768
- bos=1, eos=2, pad=0, unk=0 (all in tokenizer_config.json)
- Weight tying: `lm_head.weight` shares `embed_tokens.weight` (same Tensor object)
- MoE: when `use_moe=True`, uses routed experts + shared experts with load-balancing aux loss

## Training Pipeline (order matters)

```
train_pretrain.py → train_full_sft.py → [train_lora.py / train_dpo.py / train_reason.py] → [train_grpo.py / train_ppo.py / train_spo.py] → train_distillation.py
```

Each stage depends on the previous stage's checkpoint. Skipping stages will fail at weight loading.

## Common Gotchas

1. **`out/` files are Git LFS pointers** — real weights must be fetched with `git lfs pull`
2. **PyTorch 2.6+ `torch.load` default changed** — `.pth` files require `weights_only=False` when loading
3. **DDP `set_to_none=True`** — gradient zeroing uses `set_to_none` for speed
4. **No test suite** — validation is via `eval_llm.py` interactive testing
5. **`autocast` is autocast_ctx** — mixed precision is configured per-script via `torch.cuda.amp.autocast`
6. **Tokenizer files in `model/`** — `tokenizer.json` and `tokenizer_config.json` are the authoritative tokenizer. Do NOT retrain.
7. **Data files are gitignored** — actual `.jsonl` data lives in `minimind_dataset/` submodule
8. **ChatML format** — system messages wrapped in system/user/assistant with special tokens, not raw text

## Coding Conventions

- Chinese comments throughout (Chinese-authored educational project)
- No type hints on most functions
- No formal linting/formatting config
- snake_case throughout; model config fields use camelCase (HuggingFace convention)
- Scripts set `sys.path` manually at top rather than using package installation
- Checkpoint save: `../out/{stage}_{hidden_size}[_moe].pth`
- Checkpoint resume: `../checkpoints/` (not gitignored)

## Study Plans & Q&A Convention

The `plan/` directory contains detailed per-script study plans and a `learning_checklist.md` tracking learning progress. These are user-facing educational materials, not part of the training pipeline.

**When the user asks a question during learning, answer it AND record the answer in the corresponding study plan document** (`plan/train_*_study_plan.md`). Place the Q&A in the appropriate section — the relevant code walkthrough section or the Q&A section at the end. Follow the format: question as a bold heading, answer below. If no suitable section exists, append to the closest logical position. This dual-annotation standard (answer in conversation + append to doc) ensures the study plan stays up-to-date as a reference.
