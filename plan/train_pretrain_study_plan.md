# train_pretrain.py 学习计划指引

## 一、文件定位

`train_pretrain.py` 是 MiniMind 项目**预训练阶段**的训练脚本。它在大规模无标注文本数据上对模型进行自回归语言建模训练，让模型学会"续写"能力——即根据上文预测下一个 token。

从训练流水线来看，这是第一关：

```
train_pretrain.py（预训练，你在这里）
    ↓
train_full_sft.py（指令微调，学会对话格式）
    ↓
train_reason.py / dpo / ppo / grpo / spo（偏好对齐/推理微调）
```

## 二、前置知识

| 概念 | 建议学习途径 |
|------|-------------|
| PyTorch `DataLoader` / `Dataset` | PyTorch 官方教程 |
| 分布式训练（`DistributedDataParallel` / `DistributedSampler`） | PyTorch DDP 文档 |
| AdamW 优化器 & 学习率调度（cosine decay + warmup） | 相关博客 |
| 交叉熵损失（CrossEntropyLoss） | PyTorch 文档 |
| `torch.cuda.amp` 混合精度训练 | PyTorch AMP 教程 |
| gradient accumulation / clip | 相关博客 |

## 三、文件逐段精读

### 第 1 层：导入与全局配置（L1–L20）

```python
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, lm_checkpoint, init_distributed_mode, ...
```

**要点**：
- `PretrainDataset` 是自定义数据集，负责从 `.jsonl` 文件中读取预训练文本并 tokenize
- `init_distributed_mode` 负责初始化 DDP 分布式环境
- `lm_checkpoint` 负责保存/恢复训练 checkpoint（含 optimizer 状态）
- `SkipBatchSampler` 用于跳过某些 batch（如 checkpoint 恢复时）

### 第 2 层：`train_epoch()`（L23–L130+）

单轮训练的核心循环，包含：

- **前向传播**：`model(input_ids, labels=labels)` → loss
- **反向传播**：`loss.backward()`
- **梯度累积**：多小步累积一次 optimizer.step()
- **梯度裁剪**：防止梯度爆炸
- **学习率调度**：cosine decay + warmup
- **日志记录**：loss / lr / 速度 / ETA
- **Checkpoint 保存**：定期保存模型权重和训练状态

### 第 3 层：`main()`（L130+）

- 解析命令行参数（`--epochs`, `--batch_size`, `--lr`, `--data_path` 等）
- 初始化模型（调 `init_model()`）
- 初始化 DataLoader 和优化器
- 循环调用 `train_epoch()`

## 四、关键参数

| 参数 | 含义 |
|------|------|
| `--epochs` | 训练轮次 |
| `--batch_size` | 每张卡的 batch size |
| `--lr` | 峰值学习率 |
| `--data_path` | 预训练数据路径（.jsonl） |
| `--accumulation_steps` | 梯度累积步数 |
| `--wandb` | 是否启用 wandb 日志 |

## 五、关联文件

```
train_pretrain.py
 ├─ model/model_minimind.py     ← 模型定义
 ├─ dataset/lm_dataset.py        ← PretrainDataset 数据加载
 ├─ trainer/trainer_utils.py     ← 工具函数（lr_schedule, checkpoint, DDP init...）
 └─ trainer/train_full_sft.py    ← 下一阶段：指令微调（结构类似）
```

## 六、学习目标检查清单

- [ ] 能画出预训练的完整训练循环流程图
- [ ] 能说出 gradient accumulation 的作用和数值等价性
- [ ] 能解释为什么 pretrain 用自回归 loss（next token prediction）而不是其他 loss
- [ ] 能说明 DDP 分布式训练中 `DistributedSampler` 的作用
- [ ] 能区分 `model.train()` 和 `model.eval()` 的行为差异
- [ ] 能理解 checkpoint 保存了哪些内容以及如何恢复训练
- [ ] 能对比 `train_pretrain.py` 和 `train_full_sft.py` 的数据处理差异
