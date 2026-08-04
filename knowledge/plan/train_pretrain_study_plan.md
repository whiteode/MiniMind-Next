# train_pretrain.py 学习计划指引

## 一、文件定位

`train_pretrain.py` 是 MiniMind 项目**预训练阶段**的训练脚本。它在大规模无标注文本数据上对模型进行自回归语言建模训练，让模型学会"续写"能力——即根据上文预测下一个 token。

从训练流水线来看，这是第一关：

```
train_pretrain.py（预训练，✅ 已完成）
    ↓
train_full_sft.py（指令微调，← 下一个）
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
from scripts.Model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from scripts.Trainer.trainer_utils import get_lr, Logger, lm_checkpoint, init_distributed_mode, ...
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
 ├─ scripts/Model/model_minimind.py     ← 模型定义
 ├─ dataset/lm_dataset.py        ← PretrainDataset 数据加载
 ├─ scripts/Trainer/trainer_utils.py     ← 工具函数（lr_schedule, checkpoint, DDP init...）
 └─ scripts/Trainer/train_full_sft.py    ← 下一阶段：指令微调（结构类似）
```

## 六、学习目标检查清单

^- [x] 能画出预训练的完整训练循环流程图
      ```text
      ┌──────────────────────────────────┐
      │      每个 epoch 开始               │
      │  setup_seed(42+epoch)             │
      │  torch.randperm 打乱 indices      │
      └──────────┬───────────────────────┘
                ▼
      ┌──────────────────────────────────┐
      │  DataLoader 取一个 micro batch   │
      │  (batch_size × max_seq_len)      │
      └──────────┬───────────────────────┘
                ▼
      ┌──────────────────────────────────┐
      │  with autocast_ctx:              │ ← 混合精度：matmul→bf16，
      │    res = model(input_ids, labels) │   softmax→fp32
      │    loss = res.loss + res.aux_loss │ ← 含 MoE 辅助损失
      └──────────┬───────────────────────┘
                ▼
      ┌──────────────────────────────────┐
      │  loss = loss / accumulation_steps│ ← 梯度累积：缩放 loss
      └──────────┬───────────────────────┘
                ▼
      ┌──────────────────────────────────┐
      │  scaler.scale(loss).backward()   │ ← 如果是 fp16 模式，放大 loss
      │                                  │   后再 backward，防下溢
      └──────────┬───────────────────────┘
                ▼
      ┌──────────────────────────────────┐
      │  判断是否达到累积步数：           │
      │  (step+1) % accumulation == 0 ?  │───否──→ 继续取下一个 micro batch
      └──────────┬──是───────────────────┘
                ▼
      ┌──────────────────────────────────┐
      │  scaler.unscale_(optimizer)      │ ← 恢复梯度真实大小（fp16 模式）
      │  clip_grad_norm_(model, max_norm)│ ← 梯度裁剪，防爆炸
      │  scaler.step(optimizer)          │ ← 更新参数
      │  scaler.update()                 │ ← 更新缩放因子
      │  optimizer.zero_grad()           │ ← 清空梯度，准备下一轮累积
      └──────────┬───────────────────────┘
                ▼
      ┌──────────────────────────────────┐
      │  定期（每 save_interval 步）：    │
      │  ① model.eval()                  │
      │  ② 保存普通 .pth（仅权重 fp16）   │
      │  ③ lm_checkpoint() 保存 _resume  │
      │     （含 optimizer + scaler +    │
      │      epoch+step+indices）         │
      │  ④ model.train()                 │
      └──────────┬───────────────────────┘
                ▼
      ┌──────────────────────────────────┐
      │  epoch 结束，回到顶部开始下一轮    │
      └──────────────────────────────────┘
      ```
^- [x] 能说出 gradient accumulation 的作用和数值等价性
^- [x] 能解释为什么 pretrain 用自回归 loss（next token prediction）而不是其他 loss
      先说清一个容易混淆的点：
      "自回归 loss"不是一种独特的损失函数，损失函数就是交叉熵。
      自回归指的是**训练目标**：给定前 t-1 个 token，预测第 t 个 token。
      交叉熵衡量"预测的概率分布"和"真实 token"之间的差距。
      为什么选这个目标而不是其他（如 BERT 的掩码语言模型）？
      ① MiniMind 是生成式模型（GPT 架构），最终目标是续写文本。
         自回归预测 next token 和推理时的生成方式完全一致
         （训练和推理之间的 gap 最小）。
      ② 掩码语言模型（BERT）学的是"理解"（填空），不是"生成"（续写）。
         如果 pretrain 用 MLM，后面做文本生成还需要额外的 adaptation。
      ③ 自回归的 loss 计算简单高效：每个 token 位置都有监督信号
         （输入序列长度 L 就有 L-1 个预测目标），数据利用率高。

      【追问：MLM 和自回归的区别是什么？为什么 MLM 学的是理解而不是生成？】

      MLM（Masked Language Model，掩码语言模型）是 BERT 提出的预训练方式。

      自回归（GPT）：                 MLM（BERT）：
      输入："我今天去[上学]了"        输入："我今天去[MASK]了"
      任务：逐词预测下一个             任务：根据左右上下文猜被遮住的词
      学什么：如何接话、续写           学什么：如何理解句子结构

      举个具体例子：

      句子："苹果很好吃"

      自回归训练（GPT）：
        输入 → "苹果很好" → 预测下一个词"吃"
        模型要学会的是"苹果"之后该接什么。
        推理时：给"苹果很" → 模型接着说"好吃"。

      MLM 训练（BERT）：
        输入 → "苹果[MASK]好吃" → 预测被遮住的"很"
        模型要学会的是"苹果 ___ 好吃"这个结构中缺的是什么。
        推理时：给一个句子，模型能理解每个词在句子中的角色
        （命名实体识别、情感分类等），但不会自动续写。

      MLM 为什么学的是"理解"而不是"生成"？
      - MLM 看到的是**双向上下文**（左右的词都能看）→ 模型学会的是"这个词在句子里
        跟其他词的关系" → 这是"理解"能力的核心。
      - 自回归看到的是**单向上下文**（只看左边）→ 模型学会的是"这个词之后
        最可能接什么" → 这是"生成"能力的核心。
      - 如果你让 MLM 做生成：它不知道怎么写下去，因为它从来没学过"从左边往右边
        一个一个接词"——它学的是"从句子里填一个空"。

      MLM 转生成需要额外的 adaptation：
      比如 T5（Text-to-Text Transfer Transformer）把 MLM 改成了
      "填空式生成"——输入"苹果很[MASK]吃"，输出"好"。
      但这样仍然不是真正的自回归生成，因为每次生成一个词都要重新看一遍
      完整输入。对于长篇文本生成任务，这种方式效率极低。
      所以如果最终目标是聊天机器人或文本续写，自回归预训练是更直接的选择。
^- [x] 能说明 DDP 分布式训练中 `DistributedSampler` 的作用
      DistributedSampler（分布式采样器）负责在多卡训练时给每张卡分配不重复的数据。

      没有 DistributedSampler 时的问题：
      假设 100 条数据、2 张卡、batch_size=4：
        - 如果不做任何处理，每张卡的 DataLoader 都独立随机采样
        - 卡 0 和卡 1 可能取到相同的数据 → 梯度重复计算 → 等效 batch 虚高
        - 更糟的是：有的数据可能被多张卡取到，有的数据被所有卡跳过

      DistributedSampler 的做法：
        - 把数据集均匀分成 rank_size 份，每张卡只取自己 rank_id 对应的那份
        - 100 条数据 / 2 张卡 = 每卡 50 条，互不重叠
        - 每张卡只在这 50 条里采样，保证数据不重复

      每轮 epoch 重新 shuffle（set_epoch）：
        - 每轮 epoch 开始时调用 sampler.set_epoch(epoch)
        - 内部会根据 epoch 值重新分配每卡的数据范围
        - 这样卡 0 在 epoch 0 拿前 50 条，epoch 1 拿中间 50 条，依次轮换
        - 等价于"全数据集打乱后，按卡数等分"
^- [x] 能区分 `model.train()` 和 `model.eval()` 的行为差异
      用户回答：训练模式 dropout 开，evaluation 模式 dropout 关，norm 也会受影响。
      ✅ dropout 开关正确。
      ⚠️ "norm 也会受影响"——纠正：LayerNorm 和 RMSNorm 不受 train/eval 影响
      （它们总是在整个序列上做归一化）。会被影响的是 BatchNorm（CNN 用），
      LLM 里不用 BatchNorm。所以 LLM 的 train/eval 差异主要就是 dropout。
^- [x] 能理解 checkpoint 保存了哪些内容以及如何恢复训练
      用户回答：模型权重、优化器状态、"loss 状态"之类的都保存。
      ✅ 基本正确。
      ⚠️ "loss 状态"——纠正：loss 本身不存（每步重新算），存的是 scaler 的
      缩放因子（fp16 模式下的动态 scale）。完整内容：
        普通 .pth：model.state_dict()（fp16）
        _resume.pth：model.state_dict() + optimizer.state_dict()
          + scaler.state_dict() + epoch + step + wandb_id + indices
      恢复：把这三份 state_dict 分别 load 回 model / optimizer / scaler，
      然后从 epoch 和 step 继续训练。
^- [x] 能对比 `train_pretrain.py` 和 `train_full_sft.py` 的数据处理差异
      用户回答：pretrain 数据不特意格式化，学语义关系；SFT 数据有对话格式，
      学表达方式。
      ✅ 大方向正确。具体差异：
      PretrainDataset（lm_dataset.py:31）：
        - 数据：纯文本（jsonl 里只有 "text" 字段）
        - 处理：tokenize + truncation + padding + 全序列做 labels
        - 损失：所有 token 位置都计算 loss（整个序列都要学）
      SFTDataset（lm_dataset.py:113）：
        - 数据：对话格式（jsonl 里有 "conversations" 列表）
        - 处理：apply_chat_template → tokenize → truncation → padding
        - 损失：只有 assistant 回复部分计算 loss，prompt 部分被 mask
          （通过 generate_labels 方法将非 assisant 位置的 labels 设 -100）
        - 原因：只希望模型学会"怎么回答"，而不是"记住用户问了什么"

      【小知识：JSON vs JSONL】
      JSON：一个文件包含一个完整的 JSON 对象或数组，所有数据挤在一起。
            格式：[{"text": "A"}, {"text": "B"}]，文件结尾有 ]。
            缺点：整个文件必须一起解析，不能逐行读，文件大时内存占用高。
      JSONL（JSON Lines）：每行一个独立的 JSON 对象，行之间没有逗号和中括号。
            格式：{"text": "A"}\n{"text": "B"}\n
            优点：可以逐行读取（流式处理），每行独立，支持大文件分片处理。
            项目中 pretrain 和 SFT 的数据都是 .jsonl，就是为了方便大文件逐行读。
