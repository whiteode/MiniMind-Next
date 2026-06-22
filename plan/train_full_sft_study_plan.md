# train_full_sft.py 学习计划指引

## 一、文件定位

`train_full_sft.py` 是 MiniMind 项目**指令微调阶段**的训练脚本。在预训练模型的基础上，用高质量的对话数据让模型学会"以对话形式回答问题"。

```
train_pretrain.py（预训练，✅ 已完成）
    ↓
train_full_sft.py（指令微调，← 你在这里）
    ↓
train_reason.py / dpo / ppo / grpo / spo（偏好对齐/推理微调）
```

## 二、学习目标（和 pretrain 对比学习）

本脚本和 `train_pretrain.py` 有大量重复结构（DDP、AMP、gradient accumulation 等），
所以学习重点是**两者不同的地方**：

### 核心差异

| 方面 | train_pretrain.py | train_full_sft.py |
|------|-------------------|-------------------|
| 数据集 | PretrainDataset（纯文本） | SFTDataset（对话格式） |
| Loss 计算 | 全序列算 loss | 只对 assistant 回复算 loss |
| 损失掩码 | labels[pad] = -100 | generate_labels() 标记 prompt 为 -100 |
| 数据量 | 海量（几十 GB） | 较小（几万~几十万条） |
| 训练目标 | 学习语言规律 | 学习对话格式和指令跟随 |

### 新概念（pretrain 没有的）

1. **chat_template**：tokenizer 内置的对话模板，把 role/content 列表渲染成模型能理解的格式
   ```python
   tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
   ```

2. **Loss masking**：只有 assistant 的回复参与梯度更新，user 的 prompt 和 system prompt 被 mask
   ```python
   labels = generate_labels(input_ids)  # prompt 位置标 -100
   ```

3. **预训练权重的加载**：脚本从 `--from_weight` 参数加载 pretrain 产出 .pth 文件

## 三、文件逐段精读计划

### 第 1 层：导入与全局配置（L1–L20）
- 和 pretrain 基本一致，注意数据集换成 `SFTDataset`

### 第 2 层：`train_epoch()`（L23–L86）
- 训练循环结构和 pretrain **几乎一模一样**
- 重点理解 `lm_dataset.py:113` 中 SFTDataset 的 `__getitem__` 和 `generate_labels`
- 特别关注 `generate_labels` 的掩码逻辑：它如何识别 assistant 回复的开始和结束

### 第 3 层：`main()`（L88+）
- 重点理解 `init_model` 的 `from_weight` 参数如何加载预训练权重
- 其他参数（batch_size、lr、accumulation_steps 等）和 pretrain 含义相同

### 第 4 层：`lm_dataset.py` 中的 SFTDataset

这是和 pretrain **最本质的区别**，需要深入阅读：

```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        # 加载对话数据
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', ...).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', ...).input_ids

    def create_chat_prompt(self, conversations):
        return self.tokenizer.apply_chat_template(...)

    def generate_labels(self, input_ids):
        # 只有 <bos>assistant\n 到 <eos>\n 之间的位置算 loss
        # 其余位置（user prompt、system 等）设为 -100

    def __getitem__(self, index):
        # 调用 create_chat_prompt → tokenize → generate_labels
```

## 四、学习目标检查清单

- [ ] 能画出 SFTDataset 的数据处理流程图
- [ ] 能说出 generate_labels 的掩码策略是什么、为什么这样做
- [ ] 能解释 apply_chat_template 输出了什么格式
- [ ] 能理解 init_model 的 from_weight 如何加载 pretrain 权重
- [ ] 能对比 SFTDataset 和 PretrainDataset 的 loss 计算差异
- [ ] 能回答"为什么 SFT 只对 assistant 回复算 loss，user 的不算"
- [ ] 能在脑子里比较 train_full_sft.py 和 train_pretrain.py 的异同点

## 五、关联文件

```
train_full_sft.py
 ├─ model/model_minimind.py       ← 模型定义（和 pretrain 一样）
 ├─ dataset/lm_dataset.py          ← SFTDataset（重点学习 generate_labels）
 ├─ trainer/trainer_utils.py       ← 工具函数（和 pretrain 一样）
 └─ plan/train_pretrain_study_plan.md ← 之前的学习笔记，作为对比参考
```
