# train_reason.py 注释整理

> 本文档收录 `scripts/Trainer/train_reason.py` 中被移除的全部注释与 docstring。
> 按原代码顺序分节，每节对应原代码中的一个逻辑块 / 函数。

---

## 模块头部

> 对应原代码：第 1–23 行（导入区及模块说明）

**注释：**

```text
设置包名，并将项目根目录加入模块搜索路径，确保后续 import 能正确找到 model、dataset 等模块
```

---

## 函数 `train_epoch`

> 对应原代码：`def train_epoch():`（第 24–154 行）

**docstring：**

```text

    训练一个 epoch。
    
    该函数在推理蒸馏任务中对特殊 token（<think>、</think>、<answer>、</answer>）施加更大的 loss 权重，
    以鼓励模型学习推理过程中的结构化输出格式。

    参数:
        epoch:      当前 epoch 编号（从 0 开始）
        loader:     数据加载器
        iters:      当前 epoch 的总迭代步数（含跳过的步数）
        tokenizer:  分词器
        lm_config:  模型配置对象
        start_step: 起始步数偏移（用于断点续训时跳过已经训练过的 step）
        wandb:      wandb 日志对象（可选）
    
```

**注释：**

```text
将特殊 token 文本通过分词器编码为 token ID 列表
.input_ids 是分词器返回对象的一个字段，存放字符串对应的整数 token 序列（list[int]）
例如 tokenizer('<think>').input_ids 可能返回 [101, 205, 102]
这些整数 ID 会被模型用作输入或参与 loss 计算，后续用于对特殊 token 施加额外权重
reduction='none' 表示不对 loss 求和或求平均，保留每个 token 的 loss 值
input_ids 和 labels 由 SFTDataset（dataset/lm_dataset.py）构造：
  - input_ids: 原始对话文本经 tokenizer 编码为 token ID 序列，不足 max_length 的部分用 pad_token_id 右侧补齐
  - labels: 全量初始化为 -100，然后扫描序列，仅在 assistant 的回答区间（bos_id ~ eos_id 之间）
    将 label 设为对应位置的 token ID；其余位置（system/user 的文本、padding）保留 -100
  - 这样训练时 CrossEntropyLoss 会忽略 -100 的位置，只对 assistant 回答部分计算 loss
根据当前进度计算余弦退火学习率
前向传播
语言模型 loss 计算：预测下一个 token
语言模型的任务是"给定前文，预测下一个 token"（自回归）。
因此模型在位置 i 的 logits 应当预测位置 i+1 的真实 token。
通过 shift 对齐：将 logits 去掉最后一个位置（[:-1]），label 去掉第一个位置（[1:]），
这样 shift_logits[:, i, :] 与 shift_labels[:, i] 一一对应：
   shift_logits[:, i, :] = 模型对"第 i 个位置之后的下一个 token"的预测
   shift_labels[:, i]    = 第 i+1 个位置的真实 token ID
shift_logits: [batch, seq_len-1, vocab_size]
shift_labels: [batch, seq_len-1]
loss_fct = nn.CrossEntropyLoss(reduction='none') 定义于本函数开头
reduction='none' 表示不求和也不平均，保留每个 token 独立的 loss 值
先将 logits 展平为 [batch*(seq_len-1), vocab_size]，labels 展平为 [batch*(seq_len-1)]
计算完逐 token loss 后再 reshape 回 [batch, seq_len-1]
loss 形状 [batch, seq_len-1]，每个位置的交叉熵 loss
构造 loss 掩码：忽略 label 为 -100 的位置（即 padding 部分）
找出所有属于特殊 token 的位置（<think>、</think>、<answer>、</answer>）
对特殊 token 位置（<think>、</think>、<answer>、</answer>）的 loss 权重设为 10，其余非 padding 位置为 1
原因：推理蒸馏（reasoning distillation）的目标是让模型学会结构化思考的格式。
这些特殊 token 是定义思考/回答边界的核心标记，数量远少于普通文本 token。
增大它们的权重可以迫使优化器更关注这些关键位置的预测正确性，
否则模型很容易忽略这些稀疏但重要的标记，导致生成的思考过程格式混乱。
加权 loss 求和并归一化（按非 padding token 数做平均）
总 loss = 加权交叉熵 loss + 辅助 loss（如 MoE 的负载均衡 loss）
梯度累积：将 loss 除以累积步数，使得多个 micro-batch 的梯度平均后等效于一个全局 batch
反向传播，累积梯度
当累积步数达到 accumulation_steps 时，更新模型参数
对梯度解缩放（混合精度训练时，梯度是 scaled 的，需要 unscaled 后才能 clip）
梯度裁剪，防止梯度爆炸
优化器更新（内部会判断是否使用混合精度）
日志打印
预计剩余时间（分钟）
定期保存模型权重
从 DDP 或 torch.compile 中取出原始模型
将权重转为半精度再保存，减小存储空间
保存完整的 checkpoint（含优化器状态、scaler 状态等，用于断点续训）
```

---

## 命令行参数

> 对应原代码：`# ========================== 命令行参数 ==========================` 段（第 155–180 行）

**注释：**

```text
========================== 命令行参数 ==========================
```

---

## 1. 初始化分布式环境和随机种子

> 对应原代码：`# ========== 1. 初始化分布式环境和随机种子 ==========` 段（第 181–187 行）

**注释：**

```text
========== 1. 初始化分布式环境和随机种子 ==========
如果启动了分布式（多卡训练），自动将设备设置为当前进程对应的 GPU
不同进程使用不同随机种子，保证数据打乱不重复
```

---

## 2. 创建保存目录、初始化模型配置、检测续训 checkpoint

> 对应原代码：`# ========== 2. 创建保存目录、初始化模型配置、检测续训 checkpoint ==========` 段（第 188–193 行）

**注释：**

```text
========== 2. 创建保存目录、初始化模型配置、检测续训 checkpoint ==========
如果启用断点续训（from_resume==1），自动从 ../checkpoints 目录加载保存的训练状态
```

---

## 3. 设置混合精度上下文

> 对应原代码：`# ========== 3. 设置混合精度上下文 ==========` 段（第 194–199 行）

**注释：**

```text
========== 3. 设置混合精度上下文 ==========
CPU 下不支持 amp，使用空上下文；GPU 下启用自动混合精度
```

---

## 4. 初始化 wandb 日志（仅主进程）

> 对应原代码：`# ========== 4. 初始化 wandb 日志（仅主进程） ==========` 段（第 200–209 行）

**注释：**

```text
========== 4. 初始化 wandb 日志（仅主进程） ==========
如果从 checkpoint 恢复，尝试沿用之前的 wandb run id
```

---

## 5. 初始化模型、分词器、数据集、优化器、梯度缩放器

> 对应原代码：`# ========== 5. 初始化模型、分词器、数据集、优化器、梯度缩放器 ==========` 段（第 210–221 行）

**注释：**

```text
========== 5. 初始化模型、分词器、数据集、优化器、梯度缩放器 ==========
分布式训练时使用 DistributedSampler 自动分配数据分片
GradScaler：float16 训练时用于动态缩放 loss，防止梯度下溢
```

---

## 6. 从 checkpoint 恢复模型、优化器、scaler 状态

> 对应原代码：`# ========== 6. 从 checkpoint 恢复模型、优化器、scaler 状态 ==========` 段（第 222–230 行）

**注释：**

```text
========== 6. 从 checkpoint 恢复模型、优化器、scaler 状态 ==========
```

---

## 7. 用 DistributedDataParallel 包装模型（多卡训练）

> 对应原代码：`# ========== 7. 用 DistributedDataParallel 包装模型（多卡训练） ==========` 段（第 231–236 行）

**注释：**

```text
========== 7. 用 DistributedDataParallel 包装模型（多卡训练） ==========
忽略 RoPE 位置编码中的频率缓存，它们在各卡之间相同，无需同步
```

---

## 8. 多 epoch 训练循环

> 对应原代码：`# ========== 8. 多 epoch 训练循环 ==========` 段（第 237–252 行）

**注释：**

```text
========== 8. 多 epoch 训练循环 ==========
分布式模式下调用 set_epoch 确保每个 epoch 数据打乱方式不同
单机模式下用随机索引实现不打乱
如果要从某个 step 续训，计算需要跳过的样本数
```

---

## 9. 训练结束，销毁分布式进程组

> 对应原代码：`# ========== 9. 训练结束，销毁分布式进程组 ==========` 段（第 253–255 行）

**注释：**

```text
========== 9. 训练结束，销毁分布式进程组 ==========
```
