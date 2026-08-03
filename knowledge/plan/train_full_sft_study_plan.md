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

## 四、学习目标检查清单（含答案详解）

### □ 1. 能画出 SFTDataset 的数据处理流程图

```
原始数据（JSONL）
  [
    {"conversations": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "天气怎么样？"},
      {"role": "assistant", "content": "今天晴天"}
    ]},
    ...
  ]
         │
         ▼
  pre_processing_chat(conversations)
    对话预处理（清理格式、处理工具调用等）
         │
         ▼
  create_chat_prompt(conversations)
    tokenizer.apply_chat_template(
      messages,
      tokenize=False,         ← 先不转成 token，输出文本字符串
      add_generation_prompt=False  ← 不加生成提示（训练用，不是推理）
    )
    输出示例：
      "<|system|>\n你是一个助手\n<|user|>\n天气怎么样？\n<|assistant|>\n今天晴天\n"
         │
         ▼
  post_processing_chat(prompt)
    后处理清理（trim 空白、统一换行等）
         │
         ▼
  tokenizer(prompt).input_ids[:max_length]
    转成 token ID 序列，截断到 max_length
         │
         ▼
  input_ids += [pad_token_id] * (max_length - len(input_ids))
    padding 到固定长度（对齐 batch 内形状）
         │
         ▼
  generate_labels(input_ids)
    │
    ├── 遍历 input_ids，搜索 bos_id（即 <|assistant|>\n 的 token 序列）
    ├── 从 bos_id 之后到 eos_id 之前 → 标记为真实 token ID（参与 loss）
    └── 其余所有位置（system/user/pad）→ 标记为 -100（被 loss 忽略）
         │
         ▼
  返回 (input_ids, labels)  →  传入 model(input_ids, labels=labels) 计算 loss
```

### □ 2. 能说出 generate_labels 的掩码策略是什么、为什么这样做

**掩码策略**：

遍历 input_ids，找到每个 `<|assistant|>\n`（bos_id）出现的位置，
从 bos_id 之后开始，到 `<|end|>\n`（eos_id）之前结束，
这段区间内的 token 保留原始 ID（参与 loss 计算），
区间外所有位置（system 提示词、user 问题、padding token）全部设为 -100。

```python
labels = [-100] * len(input_ids)         # 初始全 -100
# 找到 <|assistant|>\n 之后 ~ <|end|>\n 之前
# 这个范围内的 labels[j] = input_ids[j]  # 只有助手的回答参与 loss
```

**为什么这样做**：

1. **聚焦学习目标**：SFT 的目的是让模型学会"以助手的身份回答问题"。我们不希望模型花参数去背诵用户问题或系统提示词——那是调用方在 prompt 中提供的。

2. **防止容量浪费**：如果所有 token 都算 loss，模型会用大量参数学习"如何生成用户问题"这种无关模式。词表中大部分 token 出现在 user 端，模型参数会被稀释。

3. **计算效率**：只对回答部分算 loss，减少反向传播的计算量（但收益有限，因为 forward 还是要跑完整序列）。

4. **类比**：老师只批改学生的答案，不批改题目本身。学生看了足够多的题目后，自然就学会了题目格式，但学习的重点是"怎么答"。

### □ 3. 能解释 apply_chat_template 输出了什么格式

`apply_chat_template` 是 HuggingFace tokenizer 的内置方法，
它将结构化的对话列表（messages）渲染成模型能理解的文本格式。

**输入**：
```python
messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "今天天气怎么样？"},
    {"role": "assistant", "content": "今天天气晴朗，温度25°C。"},
]
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

**输出**（对于 MiniMind 使用的 tokenizer）：
```
<|system|>
你是一个有用的助手。
<|user|>
今天天气怎么样？
<|assistant|>
今天天气晴朗，温度25°C。
<|end|>
```

关键点：
- `tokenize=False`：返回字符串而不是 token ID，方便调试
- `add_generation_prompt=False`：训练时不需要生成提示（推理时需要 True，让模型从 `<|assistant|>\n` 开始生成）
- `tokenize=True` 时直接返回 input_ids（模型实际吃的格式）
- 不同模型的 tokenizer 有不同模板（chat_template），保存在 tokenizer_config.json 中

### □ 4. 能理解 init_model 的 from_weight 如何加载 pretrain 权重

`init_model`（定义在 `trainer/trainer_utils.py`）的核心逻辑：

```python
def init_model(lm_config, from_weight, device):
    model = MiniMindForCausalLM(lm_config)           # ① 创建随机初始化的模型
    tokenizer = AutoTokenizer.from_pretrained(...)    # ② 加载 tokenizer
    if from_weight != 'none':                         # ③ 如果要加载权重
        weight_path = f'../out/{from_weight}_{lm_config.hidden_size}.pth'
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # ④ 覆盖模型参数
    return model, tokenizer
```

**具体流程**：

1. 创建一个全新的 `MiniMindForCausalLM` 实例（参数随机初始化）
2. 检查 `from_weight` 参数：
   - `from_weight='pretrain'`（默认）→ 加载 `../out/pretrain_512.pth`
   - `from_weight='full_sft'` → 加载 `../out/full_sft_512.pth`
   - `from_weight='none'` → 不加载任何权重，保持随机初始化
3. `strict=False` 允许权重不完全匹配（如增减层时忽略不存在的 key）

**from_weight vs from_resume 的区别**：
- `from_weight`：加载**预训练或其他阶段产出的权重**，只加载模型参数（.pth 文件）
- `from_resume`：加载**当前阶段的 checkpoint**（_resume.pth），包含模型 + 优化器动量 + scaler + 训练进度
- 两者可以同时使用：先 `from_weight` 初始化模型，再 `from_resume` 覆盖状态

### □ 5. 能对比 SFTDataset 和 PretrainDataset 的 loss 计算差异

| 对比维度 | PretrainDataset | SFTDataset |
|---------|----------------|------------|
| **输入格式** | 原始文本（txt/jsonl 纯文本） | 对话 JSON（含 role/content 多轮） |
| **Tokenizer** | 直接 tokenize 文本 | apply_chat_template → tokenize |
| **labels 生成** | input_ids 右移一位（`input_ids[1:] + [eos_id]`） | generate_labels() 只保留 assistant 部分 |
| **Loss 掩码** | 所有 token 都参与 loss（无 -100） | system/user/pad → -100（忽略） |
| **Padding** | 通常不需要（每条截断到固定长度） | 必须 padding（对话长度差异大） |
| **Loss 含义** | 语言模型 loss：预测每个位置的下一个词 | 对话 loss：只评估"助手的回答是否正确" |
| **Loss 大小** | 较大（~3~5，取决于数据集） | 较小（~0.5~2，仅回答部分） |

**核心差异一句话**：
PretrainDataset 想学"语言本身"（每个 token 的下一个是什么），
SFTDataset 想学"怎么回答"（只有助手的回答需要被监督）。

### □ 6. 能回答"为什么 SFT 只对 assistant 回复算 loss，user 的不算"

**直接原因**：

SFT 的训练目标不是"学会看懂用户问题"，而是"学会在给定的对话历史后给出正确的回复"。
用户问题是作为**条件**（condition）提供的，不是模型需要生成的内容。

**如果 user 部分也算 loss 会怎样**：

1. **目标混淆**：模型会花容量去学习"如何生成用户问题"，但推理时用户问题是外部输入的，模型根本不需要生成它们。这浪费了模型的参数和训练数据。

2. **分布偏移**：训练时 user 问题来自数据集中的真实分布。但推理时的 user 问题千变万化，模型学到的"生成用户问题"能力毫无用处，甚至可能干扰回答生成。

3. **梯度干扰**：user 部分的 loss 产生梯度，反向传播修改模型参数。但这些梯度指向"让模型更擅长复述用户问题"，与"让模型更擅长回答问题"的方向未必一致，可能互相抵消。

**类比**：
你学英语时做"中译英"练习——题目是中文，答案是要翻译的英文。
老师只批改你的英文翻译对不对，不会批改"你的中文题目抄对了没有"。
题目本身是输入条件，你不需要学会"出题"，只需要学会"答题"。

**那特殊 token（<|assistant|> 等）的 embedding 是怎么被训练的？**
详见上一个问题的答案——它们通过 causal attention 从回答 token 的 loss 中
"间接"获得梯度，不需要直接监督。

### □ 7. 能在脑子里比较 train_full_sft.py 和 train_pretrain.py 的异同点

**相同的部分（占代码 80% 以上）**：

| 模块 | 相同点 |
|------|--------|
| 训练循环 | `train_epoch()` 结构完全一样：取 batch → forward → loss → backward → clip → step → log → save |
| 混合精度 | `autocast_ctx` + `GradScaler` 逻辑完全相同 |
| DDP 分布式 | `init_distributed_mode` → `DistributedDataParallel` → `_ddp_params_and_buffers_to_ignore` |
| 学习率调度 | `get_lr()` 函数 + `param_group['lr'] = lr` 完全一样 |
| Checkpoint | `lm_checkpoint` 保存/恢复机制完全一样 |
| 梯度累积 | `loss / accumulation_steps` → `scaler.scale(loss).backward()` → `optimizer.step()` |
| 日志/可视化 | `Logger` + `wandb` 使用完全相同 |

**不同的部分（核心差异）**：

| 差异点 | train_pretrain.py | train_full_sft.py | 原因 |
|--------|-------------------|-------------------|------|
| **数据集** | `PretrainDataset` | `SFTDataset` | 数据格式不同（原始文本 vs 对话） |
| **Loss 范围** | 全序列 | 仅 assistant 回答 | SFT 只需要学"怎么答" |
| **Labels 生成** | input_ids 右移 | generate_labels() | 需要 loss masking |
| **学习率** | ~1e-4 | ~1e-6 | SFT 只需微调，防止遗忘 |
| **积累步数** | accumulation_steps=8 | accumulation_steps=1 | SFT 数据量小，batch 够用 |
| **训练轮数** | 多（5~10+） | 少（1~3） | SFT 数据量小，且只需微调 |
| **权重初始化** | 随机初始化 | 加载 pretrain 权重 | SFT 基于预训练模型 |
| **Padding** | 极少需要 | 必须 | 对话长度差异大 |

## 五、学习完成总结

`train_full_sft.py` 和 `train_pretrain.py` 在代码结构上高度相似（DDP、AMP、gradient accumulation、checkpoint 等机制完全复用），**核心差异只有一处：loss 计算范围**。但这一个差异带来了数据格式、掩码策略、学习率、训练轮数等一系列连锁变化。掌握了这一点，SFT 脚本就理解了 80%。

## 六、关联文件

```
train_full_sft.py
 ├─ model/model_minimind.py       ← 模型定义（和 pretrain 一样）
 ├─ dataset/lm_dataset.py          ← SFTDataset（重点学习 generate_labels）
 ├─ trainer/trainer_utils.py       ← 工具函数（和 pretrain 一样）
 └─ plan/train_pretrain_study_plan.md ← 之前的学习笔记，作为对比参考
```
