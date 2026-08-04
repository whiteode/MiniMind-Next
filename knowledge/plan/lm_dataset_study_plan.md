# lm_dataset.py 学习计划

> **文件位置**: `scripts/Dataset/lm_dataset.py`（278 行）
> **角色**: 所有训练脚本的数据管道——把原始 JSONL 文本变成模型能吃的 Tensor
> **前置知识**: 已学完 model_minimind.py、trainer_utils.py、所有 train_*.py

---

## 文件全景图

```
lm_dataset.py
│
├── 全局函数
│   ├── pre_processing_chat()     ← 给对话加 system prompt（20% 概率）
│   └── post_processing_chat()    ← 清理空 think 标签（95% 概率删除）
│
├── PretrainDataset(Dataset)      ← 预训练：纯文本 → next token prediction
├── SFTDataset(Dataset)           ← 指令微调：对话格式 → 只训练 assistant 回复
├── DPODataset(Dataset)           ← 偏好对齐：chosen/rejected 对比
└── RLAIFDataset(Dataset)         ← RLHF/GRPO/PPO/SPO：生成 prompt + answer
```

**核心使命**: 四个 Dataset 子类，分别服务于训练管线的不同阶段：

```
PretrainDataset  →  train_pretrain.py
SFTDataset       →  train_full_sft.py / train_lora.py / train_reason.py / train_distillation.py
DPODataset       →  train_dpo.py
RLAIFDataset     →  train_grpo.py / train_ppo.py / train_spo.py
```

---

## 第一章：两个全局函数——数据预处理的"小动作"

### 1.1 pre_processing_chat() —— 随机注入 system prompt

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        # ... 共 10 条中英文 system prompt
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
```

**大白话解释**:
这就像给一段没有开头的文章随机加一个"身份标签"。

```
输入 conversations（没有 system）:
[user: "你好"], [assistant: "你好！有什么可以帮你？"]

20% 概率 → 加 system:
[system: "你是一个可靠的AI"] + [user] + [assistant]

80% 概率 → 保持原样:
[user] + [assistant]
```

**设计意图**:
1. **数据增强**: 同样的对话，有时有 system prompt，有时没有——让模型学会在不同场景下工作
2. **防止过拟合**: 如果 100% 都有 system prompt，模型可能过度依赖它；随机注入让模型学会"有则用，无则跳"
3. **只对没有 system 的数据**: `conversations[0].get('role') != 'system'` 检查第一条是否已经是 system，避免重复添加

**Q: 为什么 ratio 是 0.2 而不是 0.5？**
A: 大部分真实对话没有 system prompt，0.2 保持了这个分布。太高会让模型过度依赖 system prompt。

---

### 1.2 post_processing_chat() —— 清理空 thinking 标签

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content
```

**大白话解释**:
有些 SFT 数据里，assistant 的回复是空的 `<think></think>`（没有思考内容）。这个函数**随机删除**这些空标签。

```
输入 prompt 包含: "...assistant\n<think>\n\n</think>\n\n..."

95% 概率 → 删除空标签:
"...assistant\n\n..."  （干净的回复）

5% 概率 → 保留空标签:
"...assistant\n<think>\n\n</think>\n\n..."  （保留原始格式）
```

**为什么要保留 5%？**
- 如果 100% 删除，模型永远见不到空 `<think>`，推理时遇到就懵了
- 保留少量让模型知道"空 thinking 也是合法的"
- 这和 `train_reason.py` 中 `<think>权重×10` 的设计互补

---

## 第二章：PretrainDataset —— 最简单的数据集

### 2.1 数据格式（pretrain_t2t.jsonl）

```json
{"text": "给定一段文本和关键词列表，删除文本中包含所有给定关键词的子字符串。文本：\"这是一个测试句子...\""}
```

每行就是一个纯文本字符串，没有角色标签，没有对话结构——就是一大段文字。

### 2.2 __init__ —— 加载数据

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')
```

- `load_dataset('json', ...)`: HuggingFace datasets 库，把 JSONL 文件加载成内存索引
- `split='train'`: 取全部数据作为训练集（没有验证集划分）
- **不支持流式加载**: 全部读入内存，适合数据量不大的场景

### 2.3 __getitem__ —— 核心处理流程

```python
def __getitem__(self, index):
    sample = self.samples[index]
    
    # Step 1: Tokenize（不加特殊 token）
    tokens = self.tokenizer(
        str(sample['text']), 
        add_special_tokens=False, 
        max_length=self.max_length - 2,  # 预留 BOS + EOS
        truncation=True
    ).input_ids
    
    # Step 2: 手动加 BOS + EOS
    tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
    
    # Step 3: Padding 对齐
    input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
    
    # Step 4: 转 Tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    
    # Step 5: Labels = input_ids 的克隆
    labels = input_ids.clone()
    
    # Step 6: 屏蔽 Padding 位置的 loss
    labels[input_ids == self.tokenizer.pad_token_id] = -100
    
    return input_ids, labels
```

**完整流程图**:

```
原始文本: "今天天气真好，适合出去玩"
    │
    ▼ tokenizer（不加特殊 token，截断到 max_length-2）
tokens: [1234, 567, 89, 234, 567, 890, 123]
    │
    ▼ 手动加 BOS + EOS
tokens: [1, 1234, 567, 89, 234, 567, 890, 123, 2]
    │
    ▼ Padding 到 max_length=12
input_ids: [1, 1234, 567, 89, 234, 567, 890, 123, 2, 0, 0, 0]
                                                 ↑ pad pad pad
    │
    ▼ Labels = input_ids 的克隆
labels:    [1, 1234, 567, 89, 234, 567, 890, 123, 2, 0, 0, 0]
    │
    ▼ 屏蔽 Padding 位置（label = -100）
labels:    [1, 1234, 567, 89, 234, 567, 890, 123, 2, -100, -100, -100]
```

**为什么 `max_length - 2`？**
- BOS 占 1 个位置，EOS 占 1 个位置
- 如果不预留，加上 BOS/EOS 后会超出 max_length
- 这是"预算管理"：先扣掉固定开销，剩下的才是可用空间

**为什么 labels = input_ids.clone()？**
- 自回归 LM 的任务是"预测下一个 token"
- 输入 `[BOS, 你, 好]` → 标签 `[你, 好, EOS]`
- 但 PyTorch 的 `nn.CrossEntropyLoss` 会对齐计算，所以 labels 可以和 input_ids 完全一样
- **真正的区别在 -100**: Padding 位置的 label 被设为 -100，CrossEntropyLoss 会忽略这些位置


## 第三章：SFTDataset —— 指令微调的精细控制

### 3.1 数据格式（sft_t2t.jsonl）

```json
{
  "conversations": [
    {"role": "user", "content": "What is DNS?", "reasoning_content": ""},
    {"role": "assistant", "content": "DNS is the Domain Name System..."}
  ]
}
```

每行是一个对话列表，有明确的角色标签。reasoning_content 字段是给 reasoning 模型用的（可选）。

### 3.2 __init__ —— 预计算辅助 token 序列

```python
def __init__(self, jsonl_path, tokenizer, max_length=1024):
    super().__init__()
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.samples = load_dataset('json', data_files=jsonl_path, split='train')
    self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
    self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

**关键设计**: self.bos_id 和 self.eos_id 不是单个 token，而是 token 序列！

```
self.bos_id = tokenizer('<|im_start|>assistant\n').input_ids
            = [151644, 77091, 198]    三个 token

self.eos_id = tokenizer('<|im_end|>\n').input_ids
            = [151645, 198]            两个 token
```

**为什么需要预计算？**
- generate_labels() 需要扫描整个 token 序列来定位 assistant 回复的起止位置
- 每次调用 tokenizer 去编码字符串很慢，提前算好存为 self.bos_id / self.eos_id
- 这相当于字典索引 —— 训练时只需要 O(n) 的滑动窗口匹配，不需要再调 tokenizer

### 3.3 create_chat_prompt() —— 应用 Chat Template

```python
def create_chat_prompt(self, conversations):
    messages = conversations.copy()
    tools = conversations[0]['functions'] if (
        conversations and conversations[0]['role'] == 'system'
        and conversations[0].get('functions')) else None
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
```

**apply_chat_template 做了什么？**
把 [user: '你好', assistant: '你好'] 这种对话列表，渲染成模型能理解的文本格式：

```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好<|im_end|>
```

- tokenize=False: 只渲染文本，不转成 token ID
- add_generation_prompt=False: 不加生成提示

**tools 参数**: 支持 function calling 格式，但 MiniMind 实际没用（数据里没有 functions 字段）。

### 3.4 generate_labels() —— 只训练 assistant 的回复

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)   # 默认全部忽略
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)         # assistant 回复开始
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break                        # 找到回复结束
                end += 1
            # 把 assistant 回复区域的标签设为原始 token ID
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

**核心思想**: user 的输入不参与 loss 计算，只计算 assistant 回复部分的 loss。

**为什么需要用滑动窗口匹配 bos_id？**
bos_id 是 [151644, 77091, 198] 这 3 个 token 的序列。不能简单地用 input_ids.index(bos_id[0]) 找第一个，因为：
- 对话可能有多轮（user -> assistant -> user -> assistant），每轮都要处理
- user 的消息里也可能含有 151644 这个 token，需要完整匹配 3 个 token 才能确认

**可视化 —— 一轮对话的完整流程**:

```
input_ids: [151644, 77091, 198, ...user消息..., 151645, 198,
            151644, 77091, 198, ...assistant回复..., 151645, 198, 0, 0, 0]

labels:    [-100,  -100, -100, ...-100...,        -100, -100,
            -100,  -100, -100, ...原始token...,   原始token, 原始token,
            -100, -100, -100]
             ^^^         ^^^
           system/user   assistant 区域
           区域全 -100   保留原 token ID
```

**为什么 user 区域也要 -100？**
- 模型生成 user 消息是没意义的 —— user 消息是输入，不是要模型学会说的
- 如果计算 user 部分的 loss，模型会学会预测用户说什么，导致推理时重复用户输入

### 3.5 __getitem__ —— 全流程串联

```python
def __getitem__(self, index):
    sample = self.samples[index]
    conversations = pre_processing_chat(sample['conversations'])   # 随机加 system
    prompt = self.create_chat_prompt(conversations)                 # Chat Template
    prompt = post_processing_chat(prompt)                            # 清理空 think
    input_ids = self.tokenizer(prompt).input_ids[:self.max_length]  # 直接 tokenize+截断
    input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))  # pad
    labels = self.generate_labels(input_ids)                         # 逐位置算 label
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

**关键区别 vs PretrainDataset**:
1. 这里用 tokenizer(prompt) 时没有传 add_special_tokens=False —— Chat Template 已经包含特殊 token
2. 没有手动加 BOS/EOS —— Chat Template 自己带特殊标记
3. 使用 generate_labels() 精细控制哪些位置参与 loss 计算，而不是简单 clone

**Q: "这里用 tokenizer(prompt) 时没有传 add_special_tokens=False —— Chat Template 已经包含特殊 token" 是啥意思？为啥？**

看 PretrainDataset 的对照代码就清楚了。

**PretrainDataset（手动包特殊 token）**:
```python
# 1. 先纯文本分词，禁止加特殊 token
tokens = self.tokenizer(text, add_special_tokens=False, ...).input_ids
# 2. 然后手动在头尾拼接 BOS/EOS
tokens = [bos_token_id] + tokens + [eos_token_id]
```
因为原始 `sample['text']` 就是纯文本，里面没有 `<|im_start|>` 这类特殊标记，所以必须自己加上 BOS/EOS。

**SFTDataset（chat template 自带特殊 token）**:
```python
prompt = self.create_chat_prompt(conversations)   # 这步产出的字符串已经含 <|im_start|> 和 <|im_end|>
input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
```
`create_chat_prompt` 内部调用了 `apply_chat_template()`，生成的是 ChatML 格式字符串，形如：
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
```
`<|im_start|>`（BOS token, id=1）和 `<|im_end|>`（EOS token, id=2）**已经作为文本的一部分嵌在字符串中**。当 `self.tokenizer(prompt)` 去 tokenize 时：
- Tokenizer 会自动识别这些特殊标记并映射到正确的 token ID
- 不需要 `add_special_tokens=False`——因为它本来就不需要阻止 tokenizer 加额外特殊 token（字符串里已经有）
- 不需要手动拼 BOS/EOS——格式本身就是 `<|im_start|>...<|im_end|>` 成对出现的
- 此外，这个 tokenizer 的 `tokenizer_config.json` 配置了 `"add_bos_token": false, "add_eos_token": false`，即使 `add_special_tokens=True`（默认值），tokenizer 也不会画蛇添足加多余的东西

**一句话**: Pretrain 的文本是裸文本，需要自己加头尾；SFT 的文本是 ChatML 格式字符串，特殊 token 已经"长在"字符串里了，直接 tokenize 即可。

---



## 第四章：DPODataset —— 偏好对齐的对比数据

### 4.1 数据格式（dpo.jsonl）

```json
{
  "chosen": [
    {"role": "user", "content": "How would you quantify..."},
    {"role": "assistant", "content": "A strong directorial vision..."}
  ],
  "rejected": [
    {"role": "user", "content": "How would you quantify..."},
    {"role": "assistant", "content": "I don't know..."}
  ]
}
```

每行包含两个完整的对话：
- **chosen**: 人工标注的优质回复（偏好答案）
- **rejected**: 相同的 user 输入，但 assistant 回复质量差

### 4.2 __init__

```python
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')
```

与 SFTDataset 几乎一样，多了 self.padding 后备方案（如果 pad_token_id 为 None 则用 0）。

### 4.3 generate_loss_mask() —— 逻辑与 SFTDataset.generate_labels() 几乎相同

```python
def generate_loss_mask(self, input_ids):
    loss_mask = [0] * len(input_ids)      # 0 表示忽略，1 表示参与 loss
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                loss_mask[j] = 1            # 注意：这里是 1 而不是 token ID！
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return loss_mask
```

**核心异同**:

| 对比维度 | SFTDataset.generate_labels() | DPODataset.generate_loss_mask() |
|---------|-----------------------------|-------------------------------|
| 返回值类型 | token ID（或 -100） | 0/1 布尔掩码 |
| 参与区域 | 保留 assistant 区域的 token ID | 标记为 1 |
| 忽略区域 | -100 | 0 |
| 用途 | 直接传给 CrossEntropyLoss | 传给 train_dpo.py 中的自定义 loss |

**为什么 DPO 要用布尔掩码而非 -100？**

> 首先澄清：-100 **可以用**，只是多一步转换。如果 labels 里有 -100，`dpo_loss` 里做一次 `mask = (labels != -100)` 就能拿到 0/1 掩码。DPO 的 `generate_loss_mask()` 直接返回 0/1，只是为了**省掉这一步转换**——下游计算最终需要的输入形式就是 0/1。

来看 SFT 和 DPO 的 loss 计算流程对比，就清楚为什么 0/1 更自然了。

**SFT 的标准流程（用 CrossEntropyLoss）**:

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (B*L, V)
    labels.view(-1),               # (B*L) — 其中非 assistant 位置 = -100
    ignore_index=-100              # -100 的位置自动不贡献 loss
)
# 输出：标量 loss
```

`CrossEntropyLoss` 内部做了三件事：
1. `log_softmax(logits)` —— 算每个 token 在 vocab 上的 log 概率分布
2. `nll_loss(log_probs, labels)` —— 从分布中取出 label 对应的 `-log_prob`
3. `ignore_index=-100` —— 看到 label = -100 的位置直接跳过，不参与求和
4. 最终输出一个**标量**（所有非 -100 位置的 loss 求平均）

**关键**：CrossEntropyLoss 是一个封装好的黑盒子 —— 你塞进 logits 和 labels，它吐一个标量，你**拿不到中间每一步的逐 token log_prob**。

**DPO 的流程（手动算）**:

```python
# 1. 自己算逐 token log_prob
log_probs = F.log_softmax(logits, dim=2)                     # (B, L, V)
per_token_logps = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)  # (B, L)

# 2. 用 0/1 mask 屏蔽无效位置，自己算序列平均
seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # (B, 1)
mean_log_probs = (per_token_logps * mask).sum(dim=1) / seq_lengths.squeeze()  # (B,)

# 3. 分 chosen/rejected，算相对优势
#    batch 前一半 = chosen，后一半 = rejected
pi_logratios = chosen_mean - reject_mean
ref_logratios = chosen_ref_mean - reject_ref_mean
logits = pi_logratios - ref_logratios                        # 相对优势
loss = -F.logsigmoid(beta * logits).mean()                   # 标量
```

DPO 需要：
1. 拿到**每个序列的平均 log_prob**（不是每个 token 的 loss，而是序列级别的统计量）
2. 按 **chosen/rejected 配对** 做对比（batch 前一半是 chosen，后一半是 rejected）
3. 算 **相对优势**（`π_θ/π_ref` 在 chosen/rejected 上的差值）

**为什么 -100 直接做乘法不行？**

直接用 -100 做乘法会污染数据：
```python
per_token_logps = [-0.1, -0.2, -0.3, -0.4]    # 第4个位置是 padding
mask_01         = [   1,    1,    1,    0]     # 正确：sum = -0.6, count = 3, avg = -0.2
(per_token_logps * [-100, -100, -100, 0]).sum() = 10 + 20 + 30 + 0 = 60   ← 完全错误
```
DPO 手动做 `(log_probs * mask).sum()`，mask 必须是 0/1，否则 `log_prob * (-100)` 会得到离谱的数值。

**那为什么不传 -100 进去，在 `dpo_loss` 里转一下？**

可以，完全可以。`generate_loss_mask()` 完全可以返回 -100 版本的 labels，然后 `dpo_loss` 里加一行：
```python
mask = (labels != -100).float()   # -100 → 0, 其他 → 1
```
但既然 `dpo_loss` 最终需要的就是 0/1 掩码，何必多此一举？不如直接让 `generate_loss_mask()` 返回 0/1，少一步运行时转换。这就是 DPO 选择 0/1 而非 -100 的原因：**不是不能用，是不需要绕弯路**。

**一句话总结**:

```
-100 是 CrossEntropyLoss 的 ignore_index 约定，在内部跳过指定位置。
DPO 不用 CrossEntropyLoss，它手动算序列级平均 log_prob 来做偏好对比，
计算中需要 0/1 掩码做乘法 + 除法。
-100 不是不能用（转成 0/1 就一行代码），
但 DPO 的 generate_loss_mask() 直接产 0/1 更直接，省一步转换。
```

**两种方式的完整数据流对比**:

| 步骤 | SFT (CrossEntropyLoss) | DPO (手动) |
|------|----------------------|-----------|
| 1 | logits → log_softmax (内部) | logits → log_softmax (手动) |
| 2 | gather label 对应位置 (内部) | gather label → per_token_logps (B, L) |
| 3 | label = -100 的位置跳过 (内部) | mask 0/1 与 per_token_logps 相乘后 sum + 除以长度 → (B,) |
| 4 | 输出标量 loss | 序列级均分 → 分 chosen/rejected → 算相对优势 → 标量 loss |
| mask 形式 | -100 （CrossEntropyLoss 的约定） | 0/1 （手动乘除法的需要） |

### 4.4 __getitem__ —— chosen/rejected 双通道处理

```python
def __getitem__(self, index):
    sample = self.samples[index]
    chosen = sample['chosen']
    rejected = sample['rejected']

    chosen_prompt = self.tokenizer.apply_chat_template(
        chosen, tokenize=False, add_generation_prompt=False
    )
    chosen_prompt = post_processing_chat(chosen_prompt)

    rejected_prompt = self.tokenizer.apply_chat_template(
        rejected, tokenize=False, add_generation_prompt=False
    )
    rejected_prompt = post_processing_chat(rejected_prompt)

    chosen_encoding = self.tokenizer(
        chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
    )
    rejected_encoding = self.tokenizer(
        rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
    )

    chosen_loss_mask = self.generate_loss_mask(chosen_encoding['input_ids'])
    rejected_loss_mask = self.generate_loss_mask(rejected_encoding['input_ids'])

    # Shift 处理
    x_chosen = torch.tensor(chosen_encoding['input_ids'][:-1], dtype=torch.long)
    y_chosen = torch.tensor(chosen_encoding['input_ids'][1:], dtype=torch.long)
    mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
    x_rejected = torch.tensor(rejected_encoding['input_ids'][:-1], dtype=torch.long)
    y_rejected = torch.tensor(rejected_encoding['input_ids'][1:], dtype=torch.long)
    mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

    return {'x_chosen': x_chosen, 'y_chosen': y_chosen, 'mask_chosen': mask_chosen,
            'x_rejected': x_rejected, 'y_rejected': y_rejected, 'mask_rejected': mask_rejected}
```

**返回字典而非元组**!
与 PretrainDataset / SFTDataset 返回 (input_ids, labels) 不同，DPODataset 返回 6 个 Tensor 的字典。原因：
- 需要分别跟踪 chosen/rejected 两路输入
- 每一路都需要 x（输入）、y（预测目标）、mask（哪些位置有效）
- 字典比元组更可读，方便 train_dpo.py 用 batch['x_chosen'] 索引

**shift 处理**: chosen_input_ids[:-1] 和 chosen_input_ids[1:] 构成 shifted 关系。

**六个值的含义详解**:

假设 `max_length=6`，chosen 对话为 `[user: "hi", assistant: "hello"]`，经过 chat template 后完整 token 序列为：

```
下标:    0       1       2      3       4       5
       ┌──────┬──────┬──────┬──────┬──────┬──────┐
input  │ BOS  │ user │  hi  │ EOS  │ assi │ hel  │
_ids   │<|im_ │      │      │<|im_ │      │lo    │
       │start>│      │      │end>  │      │      │
       ├──────┼──────┼──────┼──────┼──────┼──────┤
       │  1   │  3   │  45  │  2   │  88  │  102 │
       └──────┴──────┴──────┴──────┴──────┴──────┘
                                         ↑assistant 回复从这开始

loss_mask:  [0,    0,    0,    0,    1,    1]   ← assistant 区域=1
```

Shift 操作后（自回归的输入-预测对）：

```
x_chosen = input_ids[:-1] = [1,   3,   45,  2,   88 ]  ← 模型看到的前5个token
y_chosen = input_ids[1:]  = [3,   45,  2,   88,  102]  ← 每个位置"下一个"的正确token
mask_chosen = loss_mask[1:] = [0,   0,   0,   0,   1]  ← 对齐 y，只有 assistant 位置参与 loss
```

**关键**：mask 对齐的是 `y` 而非 `x`。`mask_chosen[4]=1` 表示：模型根据 `x[4]=88` 预测出 `y[4]=102` 这个 token 属于 assistant 回复，它的 log_prob 应该参与 DPO loss 计算。而 `x` 中 `[BOS, user, hi, EOS]` 区域对应的预测目标 `y` 不在 assistant 区域内，mask 为 0，这些位置不参与 loss。

**在 `train_dpo.py` 中的使用**（数据流对照）:

| 步骤 | 代码 | 做了什么 |
|------|------|---------|
| 拼 batch | `x = cat([x_chosen, x_rej])` | (B, L-1) 前半 chosen 后半 rejected |
| 拼 label | `y = cat([y_chosen, y_rej])` | (B, L-1) 同上对齐 |
| 拼 mask | `mask = cat([mask_chosen, mask_rej])` | (B, L-1) 同上对齐 |
| 模型前向 | `logits = model(x)` | (B, L-1, V) 每个位置预测下一个 token 的分布 |
| 取 log_prob | `log_probs = logits_to_log_probs(logits, y)` | (B, L-1) 每个位置在"正确答案"y上的 log_prob |
| 聚合 | `(log_probs * mask).sum(dim=1) / seq_lengths` | (B,) 每个序列的平均 log_prob，只算 assistant 区域 |
| 分 half | 前 B/2 = chosen, 后 B/2 = rejected | 送入 `dpo_loss` 计算相对优势 |

---

## 第五章：RLAIFDataset —— RL 训练的 prompt + answer 结构

### 5.1 数据格式（rlaif.jsonl）

```json
{
  "conversations": [
    {"role": "user", "content": "基于以下角色信息完成一段对话..."},
    {"role": "assistant", "content": "张明：嗨，刘琳..."},
    {"role": "user", "content": "基于以上对话提出一个问题。"},
    {"role": "assistant", "content": "张明：..."}
  ]
}
```

这里的 conversations 没有嵌套 chosen/rejected 结构，而是多轮对话。RLAIFDataset 的目的是：
- 从多轮对话中**提取最后一轮 user 之前的部分作为 prompt**
- 把最后一轮 assistant 的回复作为 answer
- 模型自己去生成回复，然后交给 Reward Model 打分

### 5.2 __init__

```python
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
```

**注意**: 和 SFTDataset 不同，这里的 bos_id / eos_id 	extbf{末尾没有 \n}。
这是因为 RLAIFDataset 的 create_chat_prompt 会自己控制格式。

### 5.3 create_chat_prompt() —— 提取 prompt + answer

```python
def create_chat_prompt(self, conversations):
    messages = []
    answer = ''
    for i, turn in enumerate(conversations):
        role = 'user' if i % 2 == 0 else 'assistant'  # 按奇偶分配角色
        messages.append({"role": role, "content": turn['content']})
        answer = turn['content']                       # 最后一轮的内容就是 answer
    prompt = self.tokenizer.apply_chat_template(
        messages[:-1],                                  # 去掉最后一轮作为 prompt
        tokenize=False,
        add_generation_prompt=True                     # 这里需要 True！
    )
    prompt = post_processing_chat(prompt)
    return prompt, answer
```

**关键设计**: 为什么用 i % 2 来判断角色？
数据中的 conversations 没有 role 字段，而是隐含的奇偶交替格式（user -> assistant -> user -> assistant）。所以：
- 偶数索引 → user
- 奇数索引 → assistant

**为什么 add_generation_prompt=True？**
这是 RLAIFDataset 与 SFTDataset 的关键区别：
- SFTDataset 训练模型去完成任务，所以用 False（assistant 回复已在数据中）
- RLAIFDataset 只给 prompt（messages[:-1]），模型需要自己生成回复，所以需要最后的 assistant 提示

```
RLAIFDataset prompt（add_generation_prompt=True）:
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
                            ← 模型从这里开始生成！
```

### 5.4 __getitem__

```python
def __getitem__(self, index):
    sample = self.samples[index]
    prompt, answer = self.create_chat_prompt(sample['conversations'])
    return {
        'prompt': prompt,    # 纯文本字符串！
        'answer': answer     # 纯文本字符串！
    }
```

**返回纯文本而非 Tensor！**
这和前面三个 Dataset 完全不同！PretrainDataset、SFTDataset、DPODataset 都在 __getitem__ 里就完成了 tokenize 并返回 Tensor，而 RLAIFDataset 返回原始文本。

为什么？因为 RL 训练（GRPO、PPO、SPO）需要模型**自己生成回复**，然后将生成的文本与 answer 做比较。所以：
1. prompt 送给 model.generate() → 模型生成文本
2. 生成的文本与 answer 对比 → 计算奖励/损失
3. tokenize 在训练脚本内部完成

### 5.5 RLAIFDataset 的使用场景

在 GRPO、PPO、SPO 中，训练脚本拿到 prompt 后：

```python
# 伪代码
prompt = batch['prompt']
# 1. tokenize
input_ids = tokenizer(prompt).input_ids
# 2. 模型生成
generated = model.generate(input_ids, max_new_tokens=max_gen_len)
# 3. 解码成文本
response = tokenizer.decode(generated)
# 4. 和 answer 比较，计算奖励
reward = reward_model(response, batch['answer'])
```

---

## 第六章：四个 Dataset 的终极对比

### 6.1 核心差异速查表

| 维度 | PretrainDataset | SFTDataset | DPODataset | RLAIFDataset |
|------|----------------|------------|------------|-------------|
| 输入数据格式 | {"text": "..."} | {"conversations": [...]} | {"chosen": [...], "rejected": [...]} | {"conversations": [...]} |
| 角色标签 | 无 | 有 user/assistant | 有 user/assistant | 无（按奇偶推导） |
| 加特殊 token | 手动 BOS+EOS | Chat Template 自带 | Chat Template 自带 | Chat Template 自带 |
| Loss 控制 | 克隆后垫 -100 | generate_labels() | generate_loss_mask() | 外部处理 |
| 返回类型 | (input_ids, labels) 元组 | (input_ids, labels) 元组 | 6 字段字典 | (prompt, answer) 纯文本 |
| 预处理函数 | 不使用 | pre + post | pre + post | pre + post |
| max_length | 512（默认） | 1024（默认） | 4096（默认） | 1024（默认） |
| Tokenize 时机 | __getitem__ 内 | __getitem__ 内 | __getitem__ 内 | 训练脚本内 |
| 对应训练脚本 | train_pretrain.py | train_full_sft/lora/reason/distillation | train_dpo.py | train_grpo/ppo/spo |

### 6.2 数据流对比

```
PretrainDataset: 纯文本 -> [BOS] + text + [EOS] -> clone labels -> pad(-100)

SFTDataset: 对话 -> Chat Template -> 找 assistant 区域 -> 其余 -100

DPODataset: chosen对话 + rejected对话 -> 分别 Chat Template -> 各自算 mask

RLAIFDataset: 多轮对话 -> 去尾取 prompt -> 返回纯文本 -> 外部生成
```

### 6.3 学习要点总结

1. **BOS/EOS 的处理方式不同**: PretrainDataset 手动加，其余由 Chat Template 管理
2. **-100 的两种用法**: PretrainDataset 屏蔽 padding，SFTDataset 屏蔽 user 输入
3. **loss_mask vs -100**: DPO 用布尔 mask（自定义 loss），SFT/Pretrain 用 -100（标准 CrossEntropyLoss）
4. **预计算 bos_id/eos_id**: 用 token 序列作为滑动窗口匹配模式，避免重复调用 tokenizer
5. **Chat Template 的 add_generation_prompt**: True 表示让模型生成，False 表示训练时已有回复
6. **两个全局函数**: pre_processing_chat 随机加 system prompt，post_processing_chat 清理空 think 标签
7. **RLAIFDataset 的特殊性**: 返回纯文本而非 Tensor，因为 RL 训练需要模型自主生成

---

## 第七章：自测问题

### Q1: PretrainDataset 为什么要手动加 BOS/EOS，而不是让 tokenizer 自动加？

**答案**:
因为需要精确控制 max_length 的预算。代码中传了 add_special_tokens=False + max_length=self.max_length-2 来预留 BOS/EOS 的位置。如果让 tokenizer 自动加 BOS/EOS，tokenizer 的行为不确定，可能导致序列长度超出 max_length。

**举一个具体的数字例子**：设 `max_length=6`，长文本 tokenize 后有 8 个 token `[t1, t2, t3, t4, t5, t6, t7, t8]`。

**场景一：让 tokenizer 自动加（add_special_tokens=True，危险）**

```python
tokens = self.tokenizer(text, add_special_tokens=True, max_length=6, truncation=True).input_ids
```

某些 tokenizer 的实现是"先截断文本再加特殊 token"，结果：
```
1. tokenize 文本 → [t1, t2, t3, t4, t5, t6, t7, t8]
2. 截断到 max_length=6 → [t1, t2, t3, t4, t5, t6]           ← 6 个了
3. 加 BOS →           [BOS, t1, t2, t3, t4, t5, t6]         ← 7 个
4. 加 EOS →           [BOS, t1, t2, t3, t4, t5, t6, EOS]    ← 8 个！😱
```

最终超出 max_length=6，后续 padding 逻辑也会错乱。

即使在正确的 tokenizer 实现中（会预留特殊 token 的预算再截断），不同版本、不同 tokenizer 类型（Fast vs Slow）的行为也可能不同 —— 存在**不确定性**。

**场景二：手动控制（PretrainDataset 实际做法，安全）**

```python
tokens = self.tokenizer(text, add_special_tokens=False, max_length=4, truncation=True).input_ids
tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
```

每一步预算都清晰：
```
1. tokenize 文本 → [t1, t2, t3, t4, t5, t6, t7, t8]
2. 截断到 max_length-2=4 → [t1, t2, t3, t4]                 ← 预留 BOS/EOS 的 2 个位置
3. 手动加 BOS →           [BOS, t1, t2, t3, t4]             ← +1
4. 手动加 EOS →           [BOS, t1, t2, t3, t4, EOS]        ← +1
                            ↑ 总长度 = 1 + 4 + 1 = 6 = max_length ✅
```

**最终长度 100% 确定 = max_length**，不依赖 tokenizer 的具体实现。

**一个生活类比（铺地板）**：

```
max_length = 6m 的房间

自动法（不放心）：
  你铺了 6m 的地板（文本），才想起门口留 BOS 的 1m、墙角留 EOS 的 1m，
  结果地板铺到隔壁房间去了。

手动法（放心）：
  你只铺了 4m 地板（max_length - 2），
  然后精确在门口留 1m（BOS）、墙角留 1m（EOS），总长正好 6m ✅
```

**一句话**：手动控制把"特殊 token 算不算在 max_length 里"这个依赖 tokenizer 实现的问题，变成了确定的数学：`max_length = 1(BOS) + (max_length - 2)(文本) + 1(EOS)`。

### Q2: SFTDataset 的 generate_labels() 和 DPODataset 的 generate_loss_mask() 有什么区别？

**答案**:
1. **返回值不同**: generate_labels 返回 token ID（或 -100），generate_loss_mask 返回 0/1 掩码
2. **用途不同**: -100 用于 PyTorch 标准 CrossEntropyLoss，0/1 用于 DPO 自定义 loss
3. **逻辑相同**: 两者的滑动窗口匹配逻辑完全一致（都是找 bos_id 到 eos_id 之间的区域）

### Q3: 为什么 RLAIFDataset 返回的是纯文本而不是 Tensor？

**答案**:
因为 RL 训练（GRPO/PPO/SPO）的流程是 prompt -> 模型生成 -> 评估，无法在数据加载阶段就完成 tokenize。模型生成的回复长度不确定，需要在生成后才进行 tokenize 和 loss 计算。而 SFT/DPO 是监督学习，输入输出都是确定的。

### Q4: pre_processing_chat 的 add_system_ratio=0.2 是什么意思？为什么不是 0.5？

**答案**:
0.2 意味着 20% 的概率给对话添加 system prompt。这是为了模拟真实场景：大部分对话没有 system prompt，少部分有。如果设成 0.5，模型会过度依赖 system prompt，导致在没有 system prompt 时表现下降。

### Q5: SFTDataset 中，如果对话有多轮（user -> assistant -> user -> assistant），generate_labels() 如何处理？

**答案**:
generate_labels() 的 while 循环会扫描整个 input_ids 序列，每遇到一个 bos_id 就标记后续区域为可训练。所以多轮对话中，所有 assistant 回复区域都会被标记为可训练，所有 user 输入区域都被标记为 -100。

### Q6: RLAIFDataset 创建 chat prompt 时为什么用 i % 2 来分配角色？

**答案**:
因为 rlaif.jsonl 的数据格式里，conversations 没有 role 字段（数据格式更简洁）。约定偶数索引是 user，奇数索引是 assistant。

### Q7: DPODataset 返回的字典中为什么要做 shift 处理（[:-1] 和 [1:]）？

**答案**:
自回归语言模型的输入-标签关系是 shifted 的：输入 tokens[:-1] 预测 tokens[1:]。DPO 计算 per_token_logps 时需要知道每个位置的 log prob 和目标 token，shift 后的数据让训练脚本可以直接使用。

---

## 附录：数据文件一览

```
minimind_dataset/
├── pretrain_t2t.jsonl        → PretrainDataset     → train_pretrain.py
├── pretrain_t2t_mini.jsonl   → PretrainDataset     → train_pretrain.py（小数据调试用）
├── sft_t2t.jsonl             → SFTDataset           → train_full_sft/lora/reason/distillation
├── sft_t2t_mini.jsonl        → SFTDataset           → 小数据调试用
├── dpo.jsonl                 → DPODataset           → train_dpo.py
├── rlaif.jsonl               → RLAIFDataset         → train_grpo/ppo/spo
├── lora_exam.jsonl           → SFTDataset           → train_lora.py（考试任务）
├── lora_identity.jsonl       → SFTDataset           → train_lora.py（身份任务）
├── lora_medical.jsonl        → SFTDataset           → train_lora.py（医疗任务）
├── agent_rl.jsonl            → RLAIFDataset         → train_grpo/ppo/spo
├── agent_rl_math.jsonl       → RLAIFDataset         → train_grpo/ppo/spo
└── images/                   → 图片数据（用于多模态，非文本训练）
```

