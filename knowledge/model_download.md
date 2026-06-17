# MiniMind 模型下载指南

## 1. 原生 .pth 权重（用于 `--weight` 选项）

**下载命令**：

```bash
# 方式一：git-lfs（推荐，自动追踪大文件）
sudo apt install git-lfs          # Ubuntu
# brew install git-lfs            # macOS
git lfs install
git clone https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch.git

# 方式二：pip 安装 modelscope 库（无需 git-lfs）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('gongjy/MiniMind2-PyTorch', cache_dir='./MiniMind2-PyTorch')"

# 将 .pth 文件复制到 ./out/ 目录
cp MiniMind2-PyTorch/*.pth ./out/
```

对应 `--weight pretrain / full_sft / reason / rlhf / ppo_actor / grpo / spo` 等选项。下载后直接运行：

```bash
python eval_llm.py --weight full_sft          # 全量指令微调
python eval_llm.py --weight pretrain           # 预训练
python eval_llm.py --weight reason --use_moe 1 # MoE 推理模型
```

## 2. HF 格式完整模型（用于 `--load_from` 选项）

**HuggingFace**：`https://huggingface.co/jingyaogong/MiniMind2`

**ModelScope**：`https://www.modelscope.cn/models/gongjy/MiniMind2`

**用途**：完整的 transformers 格式（含 config.json、tokenizer 等），自包含，不依赖 MiniMind 源码结构。

```bash
git clone https://huggingface.co/jingyaogong/MiniMind2
# 或者
git clone https://www.modelscope.cn/models/gongjy/MiniMind2

python eval_llm.py --load_from ./MiniMind2
```

## 3. 在线体验（无需下载）

- **推理模型**：[https://www.modelscope.cn/studios/gongjy/MiniMind-Reasoning](https://www.modelscope.cn/studios/gongjy/MiniMind-Reasoning)
- **常规模型**：[https://www.modelscope.cn/studios/gongjy/MiniMind](https://www.modelscope.cn/studios/gongjy/MiniMind)

## 4. 权重文件命名规则

权重文件命名格式：`{weight}_{hidden_size}[_moe].pth`

| 文件名示例 | 对应参数 |
|-----------|----------|
| `pretrain_512.pth` | `--weight pretrain --hidden_size 512` |
| `full_sft_512.pth` | `--weight full_sft --hidden_size 512` |
| `reason_640_moe.pth` | `--weight reason --hidden_size 640 --use_moe 1` |
| `rlhf_768.pth` | `--weight rlhf --hidden_size 768` |

## 5. 数据集下载（如要自行训练）

**下载命令**：

```bash
git clone https://www.modelscope.cn/datasets/gongjy/minimind_dataset.git
# 或者
git clone https://huggingface.co/datasets/jingyaogong/minimind_dataset

# 将数据文件放到 ./dataset/ 目录
cp minimind_dataset/*.jsonl ./dataset/
```

快速复现推荐下载以下文件放到 `./dataset/`：
- `pretrain_hq.jsonl`（预训练数据）
- `sft_mini_512.jsonl`（指令微调数据）
