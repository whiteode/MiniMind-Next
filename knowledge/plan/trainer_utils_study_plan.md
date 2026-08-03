# trainer_utils.py 学习计划（训练工具函数）

## 一、写在前面：这个文件是干什么的？

你已经学完了 11 个训练脚本。每个脚本开头都有这么一行：

```python
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
```

**`trainer_utils.py` 就是这些公共函数的"大本营"。** 整个文件只有 338 行，做了七件事：

```
trainer_utils.py 的七件事：
┌─────────────────────────────────────────────────────────────┐
│ 1. get_model_params()   → 打印模型参数量（含 MoE 激活参数）   │
│ 2. is_main_process()    → 判断是否主进程（DDP 多卡控制）      │
│ 3. Logger()             → 只在主进程打印日志                  │
│ 4. get_lr()             → 余弦退火学习率调度                  │
│ 5. init_distributed_mode() → DDP 分布式训练初始化            │
│ 6. setup_seed()         → 固定随机种子（可复现）              │
│ 7. lm_checkpoint()      → 断点续传（存档 / 读档）            │
│ 8. init_model()         → 加载模型 + 权重                    │
│ 9. SkipBatchSampler()   → 跳过已训练的 batch                 │
└─────────────────────────────────────────────────────────────┘
```

### 学完这篇你能回答的问题

| 问题 | 涉及的函数 |
|------|-----------|
| 为什么分布式训练时不能所有 GPU 都打印日志？ | `is_main_process` / `Logger` |
| 余弦退火学习率公式里 0.1 和 0.45 是怎么来的？ | `get_lr` |
| `init_distributed_mode` 返回的 `local_rank` 是什么？ | `init_distributed_mode` |
| 为什么存档时要先写 `.tmp` 再 `os.replace`？ | `lm_checkpoint` |
| 为什么续训必须存 optimizer 状态？ | `lm_checkpoint` |
| `world_size` 变化时 step 为什么要自动转换？ | `lm_checkpoint` |
| `init_model` 的 `strict=False` 是什么意思？ | `init_model` |
| `SkipBatchSampler` 是怎么跳过已训练数据的？ | `SkipBatchSampler` |
| 为什么要固定 `cudnn.deterministic = True`？ | `setup_seed` |

### 阅读姿势

1. **先看 Logger / is_main_process**（第二章），理解分布式训练的基本概念
2. **再看 get_lr**（第三章），理解学习率调度的数学原理
3. **然后看 lm_checkpoint**（第四章），这是最复杂的函数——断点续传
4. **最后看 init_model / SkipBatchSampler**（第五、六章）
5. **做自测**（末尾 Q&A），检验是否真的理解

### 谁在用这些函数？

```
train_pretrain.py      ──┐
train_full_sft.py      ──┤
train_lora.py          ──┤
train_dpo.py           ──┤
train_reason.py        ──┤
train_grpo.py          ──┼──→ trainer_utils.py（公共依赖）
train_ppo.py           ──┤
train_spo.py           ──┤
train_distillation.py  ──┤
train_tokenizer.py     ──┘（不使用，因为它不训练模型）
```

---

## 二、Logger / is_main_process — 分布式训练的"话筒控制"

**位置**：第 31-61 行

### 2.1 为什么需要这个？

想象你有 4 张 GPU 同时训练（DDP 模式）。每张 GPU 都是一个独立的进程，都在跑同一个训练循环。如果每个进程都 print，你会看到：

```
Epoch 1, Step 100, Loss: 2.345   ← GPU 0 打印的
Epoch 1, Step 100, Loss: 2.345   ← GPU 1 打印的（一模一样！）
Epoch 1, Step 100, Loss: 2.345   ← GPU 2 打印的
Epoch 1, Step 100, Loss: 2.345   ← GPU 3 打印的
```

4 倍的重复日志，毫无意义，还把屏幕刷爆。

**解决方案**：只让 Rank 0（主进程）打印，其他 3 个进程闭嘴。

### 2.2 代码拆解

```python
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
```

一行代码，两个条件（用 `or` 连接）：

```
条件 1: not dist.is_initialized()
  → 分布式没有初始化（单 GPU 训练）
  → 当然是主进程 → return True

条件 2: dist.get_rank() == 0
  → 分布式已初始化，当前进程的编号是 0
  → Rank 0 就是主进程 → return True

两个都不满足：
  → 分布式已初始化，但当前不是 Rank 0
  → return False（你是小弟，别说话）
```

```python
def Logger(content):
    if is_main_process():
        print(content)
```

`Logger` 就是 `print` 的"分布式安全版"——只有主进程才会真正打印。

### 2.3 DDP 基础概念

```
DDP（Distributed Data Parallel）：
┌─────────────────────────────────────────────┐
│  GPU 0 (Rank 0)  │  GPU 1 (Rank 1)  │ ...  │
│  主进程           │  工作进程         │      │
│  打印日志 ✅      │  打印日志 ❌      │      │
│  保存权重 ✅      │  保存权重 ❌      │      │
│  训练 ✅          │  训练 ✅          │      │
│  反向传播 ✅      │  反向传播 ✅      │      │
│  梯度同步 ←──────→ 梯度同步          │      │
└─────────────────────────────────────────────┘

所有进程都做训练，但只有主进程负责"对外交流"（打印、保存、记录）。
```

> 大白话：DDP 就像一个工厂有 4 条流水线同时生产。每条流水线都在干活（训练），但只有"班长"（Rank 0）负责向上级汇报（打印日志、保存权重）。如果 4 个人同时汇报同样的内容，领导会疯掉。

### 2.4 get_model_params — 参数量统计

**位置**：第 18-28 行

```python
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')
```

这个函数处理 MoE 模型的"总参数量 vs 激活参数量"：

```
MoE 模型的参数量有两个数字：

总参数量（total）：所有专家的参数都算进去
  例如：base(25M) + 4个专家(4×2M) + 共享专家(2M) = 35M

激活参数量（active）：每次推理实际用到的参数
  例如：base(25M) + 2个专家(2×2M) + 共享专家(2M) = 31M
                                ↑
                        每个 token 只激活 top_k=2 个专家

输出格式：
  MoE 模型: "Model Params: 35.00M-A31.00M"  ← 总量-激活量
  普通模型: "Model Params: 26.00M"           ← 只有一个数字
```

---

## 三、get_lr — 余弦退火学习率调度

**位置**：第 64-100 行

### 3.1 为什么需要学习率调度？

训练初期：模型什么都不懂，需要**大学习率**快速学习
训练后期：模型已经很聪明了，需要**小学习率**精细调整

如果全程用同一个学习率：
- 太大 → 后期震荡，无法收敛
- 太小 → 前期学太慢，浪费时间

**余弦退火**就是一条平滑的衰减曲线，前期降得慢，中期降得快，后期又慢下来。

### 3.2 公式拆解

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

逐步拆解：

```
第 1 步: progress = current_step / total_steps    → [0, 1] 训练进度
第 2 步: angle = π × progress                      → [0, π] 弧度
第 3 步: cos(angle)                                → [1, -1] 余弦值
第 4 步: 1 + cos(angle)                            → [2, 0] 缩放
第 5 步: 0.1 + 0.45 × (1 + cos)                   → [1.0, 0.1] 最终系数
第 6 步: lr × 系数                                  → 当前学习率
```

### 3.3 用数字走一遍

假设 `lr=1e-4, total_steps=1000`：

```
训练进度    角度       cos     1+cos    系数       学习率
0%         0°        1.0     2.0      1.00      1.0e-4  ← 全速前进
25%        45°       0.707   1.707    0.868     8.68e-5
50%        90°       0.0     1.0      0.55      5.5e-5  ← 降到一半多
75%        135°     -0.707   0.293    0.232     2.32e-5
100%       180°     -1.0     0.0      0.10      1.0e-5  ← 保留 10% 底线
```

### 3.4 为什么是 0.1 和 0.45？

```
系数 = 0.1 + 0.45 × (1 + cos)

训练开始: 0.1 + 0.45 × 2 = 1.0   ← 系数 = 100%（全速）
训练结束: 0.1 + 0.45 × 0 = 0.1   ← 系数 = 10%（保留底线）

0.1 = 最低学习率比例（底线）
0.45 = 调节幅度（0.45 × 2 = 0.9，加上底线 0.1 = 1.0）
```

**为什么保留 10% 底线而不降到 0？**

如果学习率降到 0，模型参数就不再更新了——等于"学完了但还在假装学习"。保留 10% 确保模型在训练末期还能做微小的调整。

### 3.5 学习率曲线图

```
学习率
  ↑
1.0│━━━━━━━━━╲
   │          ╲
   │           ╲
   │            ╲
   │             ╲
0.5│              ╲
   │               ╲
   │                ╲
   │                 ╲━━━━━━━━
0.1│                        ━━━━ ← 10% 底线
   └──────────────────────────→ 训练步数
   0%                        100%
```

> 大白话：学习率就像开车的速度。刚上路（训练初期）要开快点赶路，到了目的地附近（训练后期）要减速慢慢找停车位。余弦退火就是一条"先快后慢"的减速曲线，但最后不会完全停车（保留 10% 速度），万一还要微调呢。

### 3.6 在训练脚本中怎么用

```python
# train_pretrain.py 中的用法
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    # 计算当前学习率
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

    # 手动设置到优化器
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

注意：这里没有用 PyTorch 的 `LR Scheduler`，而是**手动计算 + 手动设置**。这样更灵活，可以精确控制每一步的学习率。

---

## 四、lm_checkpoint — 断点续传（最复杂的函数）

**位置**：第 121-297 行

### 4.1 为什么需要断点续传？

训练大模型可能需要几天甚至几周。如果训练到一半：

```
训练到一半突然：
  - 电脑断电了 💥
  - GPU 过热宕机了 🔥
  - 实验室网络断了 🌐
  - 你不小心关了终端 😅

没有断点续传：
  → 从头开始训练 → 浪费几天的算力 💸

有断点续传：
  → 从上次中断的地方继续 → 只浪费最后几分钟 ⏱️
```

### 4.2 两种文件：普通权重 vs 续训存档

```
lm_checkpoint 生成两种文件：

1. 普通权重文件: pretrain_512.pth
   ┌─────────────────────────────┐
   │ model.state_dict()          │  ← 只有模型参数
   │ 经过 .half() 处理           │  ← 半精度，文件小
   │ 适合部署推理                 │  ← 推荐用这个
   └─────────────────────────────┘
   大小: ~13MB（MiniMind Small）

2. 续训存档文件: pretrain_512_resume.pth
   ┌─────────────────────────────┐
   │ model.state_dict()          │  ← 模型参数
   │ optimizer.state_dict()      │  ← 优化器动量（最大头！）
   │ epoch / step                │  ← 训练进度
   │ world_size                  │  ← GPU 数量
   │ wandb_id                    │  ← 实验 ID
   │ **kwargs                    │  ← 其他状态
   └─────────────────────────────┘
   大小: ~40MB（MiniMind Small，约 3 倍）
   仅用于断点续传，不用于部署
```

### 4.3 存档模式（model is not None）

```python
if model is not None:  # 存档模式
```

流程图：

```
输入: model, optimizer, epoch, step, wandb
  │
  ▼
① 脱掉"外套"（DDP / torch.compile）
  raw_model = model.module                    # 脱掉 DDP 外套
  raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 脱掉 compile 外套
  │
  ▼
② 保存普通权重（小文件）
  state_dict = raw_model.state_dict()
  state_dict = {k: v.half().cpu() for ...}    # 转半精度 + CPU
  写入 .tmp → os.replace → ckp_path          # 原子写入
  │
  ▼
③ 保存续训存档（大文件）
  resume_data = {
      'model': state_dict,                    # 模型参数
      'optimizer': optimizer.state_dict(),    # 优化器状态
      'epoch': epoch,                         # 轮次
      'step': step,                           # 步数
      'world_size': ...,                      # GPU 数量
      'wandb_id': wandb_id                    # 实验 ID
  }
  遍历 **kwargs，存入额外状态
  写入 .tmp → os.replace → resume_path       # 原子写入
  │
  ▼
④ 清理显存
  del state_dict, resume_data
  torch.cuda.empty_cache()
```

### 4.4 为什么要先写 `.tmp` 再 `os.replace`？

```
直接写入（危险）：
  torch.save(data, "checkpoint.pth")
  → 写到一半断电 → 文件损坏 → 旧的也没了 → 全部白费 💀

原子写入（安全）：
  torch.save(data, "checkpoint.pth.tmp")  ← 先写临时文件
  os.replace("checkpoint.pth.tmp", "checkpoint.pth")  ← 原子替换
  → 写到一半断电 → .tmp 损坏，但旧的 .pth 还在 → 损失可控 ✅
```

`os.replace` 在大多数文件系统上是**原子操作**——要么完成，要么不发生，不会出现"写了一半"的状态。

### 4.5 为什么要存 optimizer 状态？

```
AdamW 优化器为每个参数维护两个"动量"：
  - m: 一阶动量（梯度的移动平均）→ 告诉你"梯度方向"
  - v: 二阶动量（梯度平方的移动平均）→ 告诉你"梯度大小"

如果续训时不存 optimizer：
  → m 和 v 全部归零
  → 模型失去了"惯性"
  → Loss 会瞬间跳变（Spike），可能训飞

如果存了 optimizer：
  → m 和 v 恢复到中断前的状态
  → 训练平滑继续，Loss 曲线接上
```

> 大白话：optimizer 的动量就像火车的惯性。火车开着开着突然停下来（中断），如果重新启动时没有惯性（不存 optimizer），就要从静止开始加速——会猛抖一下（Loss Spike）。如果存了惯性，重新启动时直接接上之前的速度，平滑继续。

### 4.6 读档模式（model is None）

```python
else:  # 读档模式
    if os.path.exists(resume_path):
        ckp_data = torch.load(resume_path, map_location='cpu')
        saved_ws = ckp_data.get('world_size', 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        if saved_ws != current_ws:
            ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
        return ckp_data
    return None
```

**world_size 变化时的 step 转换**：

```
场景：之前用 2 张 GPU 训练，现在换 4 张 GPU

之前: world_size=2, 每张 GPU 处理 32 条数据
  → 总共处理了 2 × 32 × 100步 = 6400 条数据

现在: world_size=4, 每张 GPU 处理 32 条数据
  → 每步处理 4 × 32 = 128 条数据
  → 要处理 6400 条数据，只需要 6400/128 = 50 步

转换公式:
  new_step = old_step × old_world_size // new_world_size
  50 = 100 × 2 // 4
```

### 4.7 wandb_id 的作用

```python
wandb_id = None
if wandb:
    if hasattr(wandb, 'get_run'):
        run = wandb.get_run()
        wandb_id = getattr(run, 'id', None) if run else None
    else:
        wandb_id = getattr(wandb, 'id', None)
```

```
没有 wandb_id：
  每次运行脚本 → WandB 认为是新实验 → Loss 曲线从 0 重新画
  网页上看到: Run 1 (0-100步), Run 2 (0-50步)  ← 断开的

有 wandb_id + resume='must':
  续训时传入 id=wandb_id → WandB 发现 ID 匹配 → 数据追加到旧曲线
  网页上看到: Run 1 (0-100步, 50-150步)  ← 连续的
```

> 大白话：wandb_id 就像游戏的"存档位编号"。没有编号，每次开新档；有了编号，续训时能接上之前的存档，Loss 曲线就是连续的。

---

## 五、init_model — 加载模型 + 权重

**位置**：第 300-312 行

### 5.1 代码拆解

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 2. 创建空模型（随机初始化权重）
    model = MiniMindForCausalLM(lm_config)

    # 3. 加载预训练权重（如果 from_weight != 'none'）
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # 4. 打印参数量
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    return model.to(device), tokenizer
```

### 5.2 strict=False 是什么意思？

```python
model.load_state_dict(weights, strict=False)
```

`load_state_dict` 有两个模式：

```
strict=True（默认）：
  要求权重文件的 key 和模型的 key 完全一致
  多一个 key → 报错
  少一个 key → 报错

strict=False（宽松模式）：
  多的 key → 忽略
  少的 key → 用随机初始化
  不完全匹配 → 尽量加载能匹配的
```

**为什么这里用 `strict=False`？**

因为不同训练阶段的模型可能有不同的结构：

```
pretrain → full_sft:  结构相同，strict=True 也行
full_sft → lora:      加了 LoRA 层，strict=True 会报错
full_sft → reason:    可能加了特殊 token，strict=True 会报错
```

用 `strict=False` 可以灵活应对各种情况。

### 5.3 权重文件名命名规则

```
{save_dir}/{from_weight}_{hidden_size}[_moe].pth

示例：
  pretrain_512.pth          ← 预训练，Small，非 MoE
  full_sft_768.pth          ← SFT，Base，非 MoE
  grpo_640_moe.pth          ← GRPO，MoE 配置
  reason_512.pth            ← Reason，Small
```

> 大白话：init_model 就像"先搭好框架（创建模型），再往里面填水泥（加载权重）"。`strict=False` 就是"水泥多了就堆旁边，少了就空着"——灵活但不精确。

---

## 六、SkipBatchSampler — 跳过已训练的 batch

**位置**：第 315-338 行

### 6.1 为什么需要跳 batch？

续训场景：你已经训练了 500 个 batch，现在要从第 501 个继续。

```
不跳 batch：
  续训 → 从第 1 个 batch 开始 → 重复训练已学过的内容 → 浪费时间

跳 batch：
  续训 → 跳过前 500 个 batch → 从第 501 个开始 → 接着学
```

### 6.2 代码拆解

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler       # 原始的数据采样器
        self.batch_size = batch_size # 每个 batch 的大小
        self.skip_batches = skip_batches  # 要跳过的 batch 数量

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []       # 攒够一个 batch，但跳过它
                    continue
                yield batch          # 没到跳过数量，正常输出
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch              # 最后不满一个 batch 的处理

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
```

### 6.3 执行流程图

```
假设: 总共 10 个 batch，skip_batches=3

batch 编号:  [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]
操作:        跳  跳  跳  输出 输出 输出 输出 输出 输出 输出
             ↓  ↓  ↓
             丢弃 丢弃 丢弃

实际输出: batch 4, 5, 6, 7, 8, 9, 10（共 7 个）
__len__ 返回: 10 - 3 = 7
```

### 6.4 在训练脚本中怎么用

```python
# train_pretrain.py 中
if args.start_step > 0:
    # 续训：跳过已训练的 batch
    train_sampler = SkipBatchSampler(train_sampler, args.batch_size, skip_batches=args.start_step)
```

#### Q: batch 抽取不是随机的吗？跳过前 N 个 batch 为什么能避免"重复"？

好问题。如果每次 shuffle 的顺序不同，跳过前 N 个 batch 确实不能精确避免看到相同的数据。但 MiniMind 有一个**关键设计**解决了这个问题：**保存 + 恢复 `indices`**。

**核心机制**

```python
# 第一次训练时（L2768）：
setup_seed(42 + epoch)
indices = torch.randperm(len(train_ds)).tolist()  # 生成固定的随机排列

# 存档时（通过 lm_checkpoint 的 kwargs）：
lm_checkpoint(..., indices=indices)  # 把排列存进 _resume.pth

# 续训时（L2765-2766）：
if epoch == start_epoch and saved_indices is not None:
    indices = saved_indices  # 恢复上次的排列！
else:
    setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
```

**完整流程图**

```
第一次训练（epoch 0）：
  setup_seed(42) → indices = [7, 3, 1, 9, 5, 2, 8, 4, 6, 0]
  
  batch 1: [7,3,1]
  batch 2: [9,5,2]
  batch 3: [8,4,6] ← 训练到 step 3 时断了
  
  存档: indices=[7,3,1,9,5,2,8,4,6,0] 被保存到 _resume.pth


续训（epoch 0 继续）：
  从 _resume.pth 恢复: indices = [7,3,1,9,5,2,8,4,6,0]  ← 完全一样的排列！
  
  SkipBatchSampler 跳过前 3 个 batch：
  batch 1: [7,3,1] → 跳过 ✅（和第一次一样）
  batch 2: [9,5,2] → 跳过 ✅（和第一次一样）
  batch 3: [8,4,6] → 跳过 ✅（和第一次一样）
  batch 4: [0,...]  → 开始训练 ← 从断点继续！
```

**如果 `indices` 没有被保存会怎样？**

```
第一次: indices = [7, 3, 1, 9, 5, 2, 8, 4, 6, 0]（随机）
续训:   indices = [2, 8, 5, 1, 7, 9, 3, 0, 4, 6]（不同的随机）

跳过前 3 个 batch：
  第一次 batch 3: [8, 4, 6]
  续训 batch 3:   [1, 7, 9]  ← 完全不同的数据！
  
跳过没有意义，而且浪费了数据
```

**SkipBatchSampler 本身不知道"跳过的是不是相同的数据"**——它只是机械地跳过前 N 个 batch。真正保证"跳过的是相同数据"的，是 `indices` 的保存和恢复机制。

> 大白话：SkipBatchSampler 就像一个"跳过前 N 页"的书签。但它不知道第 N 页的内容是什么——它只是机械地翻过去。真正保证"跳过的是同一页"的，是你每次都在同一本书的同一个位置夹书签（`indices` 保存和恢复）。如果换了本书（`indices` 不同），书签跳过的内容就完全不一样了。

---

## 七、init_distributed_mode — DDP 分布式初始化

**位置**：第 102-109 行

### 7.1 代码拆解

```python
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式

    dist.init_process_group(backend="nccl")  # 初始化通信后端
    local_rank = int(os.environ["LOCAL_RANK"])  # 获取本地 GPU 编号
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的 GPU
    return local_rank
```

### 7.2 环境变量

DDP 由 `torchrun` 启动，`torchrun` 会自动设置这些环境变量：

```
RANK:        全局进程编号（0, 1, 2, ..., 7）
LOCAL_RANK:  本机上的进程编号（0, 1, 2, 3，如果只有 1 台机器）
WORLD_SIZE:  总进程数（= GPU 总数）
```

```
假设 2 台机器，每台 4 张 GPU：

机器 1:
  RANK=0, LOCAL_RANK=0  → GPU 0
  RANK=1, LOCAL_RANK=1  → GPU 1
  RANK=2, LOCAL_RANK=2  → GPU 2
  RANK=3, LOCAL_RANK=3  → GPU 3

机器 2:
  RANK=4, LOCAL_RANK=0  → GPU 0  ← 注意 LOCAL_RANK 从 0 开始！
  RANK=5, LOCAL_RANK=1  → GPU 1
  RANK=6, LOCAL_RANK=2  → GPU 2
  RANK=7, LOCAL_RANK=3  → GPU 3
```

### 7.3 nccl 后端

```
PyTorch 支持多种通信后端：

nccl:  NVIDIA GPU 专用，最快 ← MiniMind 用这个
gloo:  CPU / 通用，较慢
mpi:   MPI 通信库

MiniMind 只用 GPU 训练，所以选 nccl。
```

#### Q: NCCL / Gloo / MPI 三种通信后端有什么区别？

| 后端 | 适用场景 | 速度 | 依赖 |
|------|---------|:----:|------|
| **NCCL** | NVIDIA GPU ↔ GPU | 最快 | NVIDIA 驱动 + CUDA |
| **Gloo** | CPU ↔ CPU，或 GPU ↔ CPU（通用） | 慢 | 无特殊依赖 |
| **MPI** | 超级计算机 / 多机多卡 | 中等 | 需安装 MPI 库 |

**NCCL（最快）**

NVIDIA 官方专门为 GPU 间通信开发的库。直接走 GPU 显存 → PCIe/NVLink → GPU 显存，不经过 CPU。

```
GPU 0 显存 ──NVLink/PCIe──→ GPU 1 昔存
     ↑                            ↑
  直接传输，不经过 CPU 和系统内存

延迟: ~1-10 μs
带宽: 600 GB/s (NVLink), 32 GB/s (PCIe 4.0)
```

**Gloo（最慢）**

Google 开发的通用通信库。GPU 间通信需要经过 CPU 中转。

```
GPU 0 显存 → CPU 内存 → TCP/共享内存 → CPU 内存 → GPU 1 显存
     ↑                                                  ↑
  多了两次 CPU 中转，延迟高、带宽低

延迟: ~10-100 μs
带宽: ~10 GB/s (共享内存), ~1 GB/s (TCP)
```

**MPI（中等）**

超级计算机的标准通信协议。需要安装 OpenMPI 或 MPICH。在 InfiniBand 网络上效率很高。

```
GPU 0 显存 → CPU 内存 → MPI 通信 → CPU 内存 → GPU 1 显存
在 InfiniBand 上可以绕过部分 CPU 中转

延迟: ~5-50 μs
带宽: 取决于网络（InfiniBand 可达 200+ Gb/s）
```

**实际选择建议**

| 你的情况 | 选什么 |
|---------|--------|
| 有 NVIDIA GPU，单机多卡 | NCCL（唯一正确选择） |
| 有 NVIDIA GPU，多机多卡 | NCCL |
| 只有 CPU | Gloo |
| 没有 NVIDIA GPU（AMD/华为） | Gloo |
| 超级计算机集群 | MPI 或 NCCL |
| 调试/测试，不想装 NCCL | Gloo |

> 大白话：NCCL 就像"GPU 之间的高速公路"——数据直接从一张显卡传到另一张，不经过 CPU 这个"收费站"，所以最快。Gloo 就像"走国道"——数据要先下高速（GPU→CPU），走普通公路（TCP/共享内存），再上高速（CPU→GPU），绕了一大圈。MPI 就像"跨省高铁"——比国道快，但还是不如 GPU 直连的高速公路。MiniMind 只用 GPU，当然走高速公路（NCCL）。

### 7.4 单 GPU 时的行为

```python
if int(os.environ.get("RANK", -1)) == -1:
    return 0  # 非 DDP 模式
```

如果不通过 `torchrun` 启动（直接 `python train_pretrain.py`），`RANK` 环境变量不存在，返回 0。后续所有 `is_main_process()` 都返回 True，`Logger` 正常打印。

---

## 八、setup_seed — 固定随机种子

**位置**：第 112-119 行

### 8.1 代码拆解

```python
def setup_seed(seed: int):
    random.seed(seed)                  # Python 内置随机数
    np.random.seed(seed)               # NumPy 随机数
    torch.manual_seed(seed)            # PyTorch CPU 随机数
    torch.cuda.manual_seed(seed)       # PyTorch 当前 GPU 随机数
    torch.cuda.manual_seed_all(seed)   # PyTorch 所有 GPU 随机数
    torch.backends.cudnn.deterministic = True   # cuDNN 确定性算法
    torch.backends.cudnn.benchmark = False      # 关闭自动调优
```

### 8.2 为什么要固定这么多地方的种子？

```
模型训练涉及多层随机性：

1. 数据加载:  shuffle 随机打乱顺序 → random.seed
2. 权重初始化:  nn.Linear 的随机初始化 → torch.manual_seed
3. Dropout:     随机丢弃神经元 → torch.manual_seed
4. 数据增强:    随机裁剪/旋转 → numpy.random.seed
5. cuDNN:       底层库的随机算法 → cudnn.deterministic

任何一层的随机性不同 → 训练结果不同 → 无法复现
```

### 8.3 deterministic=True 和 benchmark=False

```
cudnn.deterministic = True:
  强制 cuDNN 使用确定性算法
  同样的输入 → 同样的输出（每次都一样）
  代价：可能稍微慢一点

cudnn.benchmark = False:
  关闭 cuDNN 的自动算法选择
  benchmark=True 时，cuDNN 会尝试多种算法选最快的
  但不同算法可能产生不同结果 → 不确定性
  关闭它 → 结果完全确定
```

> 大白话：固定种子就像给模型训练加了一个"录像回放"功能——同样的种子、同样的数据、同样的代码，训练结果一定一模一样。这对于调试和论文复现至关重要。

---

## 九、代码结构总览

```
trainer_utils.py（338 行）
│
├── get_model_params（L18-28）        打印参数量（含 MoE 激活参数）
├── is_main_process（L31-56）         判断是否主进程
├── Logger（L59-61）                  只在主进程打印
├── get_lr（L64-100）                 余弦退火学习率调度
├── init_distributed_mode（L102-109） DDP 初始化
├── setup_seed（L112-119）            固定随机种子
├── lm_checkpoint（L121-297）         断点续传（存档/读档）
├── init_model（L300-312）            加载模型 + 权重
└── SkipBatchSampler（L315-338）      跳过已训练的 batch
```

### 调用关系图

```
train_pretrain.py
    │
    ├── init_distributed_mode()     ← 启动时初始化 DDP
    ├── setup_seed(seed)            ← 启动时固定种子
    ├── init_model(lm_config, ...)  ← 加载模型和权重
    │
    ├── train_epoch()
    │   ├── get_lr(...)             ← 每步计算学习率
    │   ├── Logger(...)             ← 打印训练日志
    │   └── lm_checkpoint(...)      ← 每 N 步存档
    │
    └── SkipBatchSampler(...)       ← 续训时跳 batch
```

---

## 十、检查你是否真的理解（Q&A）

### 基础

**1. `is_main_process()` 的两个条件分别覆盖什么场景？**

答案：条件 1 `not dist.is_initialized()` 覆盖单 GPU 场景——没有启动分布式训练，当前进程就是唯一的进程，当然是主进程。条件 2 `dist.get_rank() == 0` 覆盖多 GPU 场景——分布式已初始化，Rank 0 被约定为主进程，负责打印、保存等"对外交流"工作。

**2. `get_lr` 公式里 0.1 和 0.45 分别是什么含义？**

答案：0.1 是最低学习率比例（底线），确保训练结束时学习率不会归零，模型还能做微小调整。0.45 是调节幅度，乘以 `(1+cos)` 的范围 [0, 2] 后得到 [0, 0.9]，加上底线 0.1 后总范围是 [0.1, 1.0]，即学习率从 100% 衰减到 10%。

**3. `setup_seed` 为什么要同时固定 random、numpy、torch、cuda、cudnn 五个地方的种子？**

答案：模型训练涉及多层随机性——数据 shuffle 用 random、数据增强用 numpy、权重初始化和 Dropout 用 torch、cuDNN 底层有自己的随机算法。任何一层的随机性不同都会导致训练结果不同。固定所有地方的种子才能确保完全可复现。

### 深入

**4. `lm_checkpoint` 为什么先写 `.tmp` 再 `os.replace`？**

答案：防止"原子性问题"——如果直接写入 checkpoint.pth，写到一半断电会导致文件损坏（旧的已被覆盖，新的没写完）。先写到临时文件 checkpoint.pth.tmp，再用 `os.replace` 原子替换，确保要么完整保存，要么保留旧文件。

**5. 续训时为什么要存 optimizer 状态？不存会怎样？**

答案：AdamW 优化器为每个参数维护一阶动量 m 和二阶动量 v，记录了梯度的历史信息。如果续训时不存 optimizer，m 和 v 归零，模型失去"惯性"，Loss 会瞬间跳变（Spike），可能训飞。存了 optimizer 能让训练平滑继续。

**6. `world_size` 从 2 变成 4 时，`step` 为什么要乘以 `saved_ws // current_ws`？**

答案：world_size 变大意味着每步处理的数据量变多了。之前 2 张 GPU 每步处理 2×batch_size 条数据，现在 4 张 GPU 每步处理 4×batch_size 条数据。为了保持总数据处理量不变，步数要按比例缩小：`new_step = old_step × old_world_size // new_world_size`。

**7. `init_model` 的 `strict=False` 有什么好处？**

答案：`strict=False` 允许权重文件的 key 和模型的 key 不完全匹配——多的 key 忽略，少的 key 用随机初始化。这在不同训练阶段切换时很有用：比如从 full_sft 切换到 lora 时，模型多了 LoRA 层，`strict=False` 能自动跳过这些不匹配的参数，避免报错。

**8. `SkipBatchSampler.__len__` 为什么要用 `max(0, ...)`？**

答案：如果 `skip_batches >= total_batches`（跳过的比总数还多），`total_batches - skip_batches` 会变成负数。`max(0, ...)` 确保返回值不为负，避免 DataLoader 报错。这在极端情况下（比如续训步数超过总步数）是一个安全兜底。

---

## 十一、动手练习

### 基础

**练习 1：手动计算学习率**

假设 `lr=2e-4, total_steps=2000`，手动计算 step=0, 500, 1000, 1500, 2000 时的学习率，画出表格。

**练习 2：验证 Logger 的分布式行为**

```python
# 在单 GPU 模式下运行
from trainer.trainer_utils import is_main_process, Logger
print(f"is_main_process: {is_main_process()}")  # 应该是 True
Logger("Hello from rank 0")  # 应该打印
```

### 进阶

**练习 3：模拟断点续传**

```python
# 模拟存档
import torch
from trainer.trainer_utils import lm_checkpoint

# 创建一个简单模型
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 存档
lm_checkpoint(lm_config=None, weight='test', model=model, 
              optimizer=optimizer, epoch=1, step=100)

# 读档
data = lm_checkpoint(lm_config=None, weight='test', model=None)
print(data.keys())  # 应该有 model, optimizer, epoch, step, world_size, wandb_id
```

**练习 4：理解 SkipBatchSampler 的行为**

```python
from torch.utils.data import DataLoader, SequentialSampler
from trainer.trainer_utils import SkipBatchSampler

# 创建一个有 100 个样本的数据集
dataset = list(range(100))
sampler = SequentialSampler(dataset)

# 跳过前 5 个 batch（batch_size=10）
skip_sampler = SkipBatchSampler(sampler, batch_size=10, skip_batches=5)
print(f"总 batch 数: {len(skip_sampler)}")  # 应该是 5（10 - 5）

# 验证输出
batches = list(skip_sampler)
print(f"第一个 batch: {batches[0]}")  # 应该是 [50, 51, ..., 59]
```

### 深入

**练习 5：对比 `.pth` 和 `_resume.pth` 的文件大小**

用 `train_pretrain.py` 训练几个 epoch，分别查看 `pretrain_512.pth` 和 `pretrain_512_resume.pth` 的文件大小，验证"resume 文件约 3 倍大"的说法。

**练习 6：修改 lm_checkpoint 支持保存 scaler 状态**

在 `lm_checkpoint` 的 `**kwargs` 中传入一个 `torch.cuda.amp.GradScaler` 对象，验证它能被正确保存和加载（提示：检查 `resume_data` 中是否多了 `scaler` key）。

#### Q: GradScaler 是干嘛的？

GradScaler 是 PyTorch **混合精度训练（AMP）** 的核心组件，用来解决 FP16 训练时的**梯度下溢（gradient underflow）**问题。

**背景：为什么需要混合精度？**

```
FP32（单精度）：  每个参数 4 字节，精度高，但慢、占显存
FP16（半精度）：  每个参数 2 字节，速度快、省显存，但精度低

混合精度训练：
  前向传播：用 FP16 计算（快）
  反向传播：用 FP32 梯度（稳）
  → 兼得速度和稳定性
```

**问题：FP16 的梯度可能太小**

```
FP16 能表示的最小正数: ~6×10⁻⁸

假设梯度值: 1×10⁻⁹（很小但有意义）
FP16 下: 被截断为 0 → 梯度消失了 → 参数不更新 → "下溢"
```

**GradScaler 的解决方案：动态缩放**

```
核心思想：把梯度"放大"到 FP16 能安全表示的范围，算完再"缩小"回去。

步骤 1: 把 loss 乘以缩放因子（比如 1024）
  scaled_loss = loss × 1024

步骤 2: 反向传播，得到放大后的梯度
  原本 1×10⁻⁹ 的梯度 → 变成 1×10⁻⁶（FP16 安全范围）

步骤 3: 更新参数（在 FP32 下）

步骤 4: 把梯度除以 1024，恢复原始尺度
```

**动态调整机制**

```
如果连续 N 步没有出现 inf/nan（训练稳定）：
  → 把缩放因子 ×2（放大更多，更激进）

如果某一步出现了 inf/nan（梯度爆炸）：
  → 跳过这一步的参数更新
  → 把缩放因子 ÷2（缩小，更保守）
```

**代码中怎么用**

```python
scaler = torch.cuda.amp.GradScaler()

for input_ids, labels in loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():       # 自动用 FP16 计算
        output = model(input_ids)
        loss = criterion(output, labels)
    scaler.scale(loss).backward()          # 放大 loss，反向传播
    scaler.step(optimizer)                 # 更新参数（如果梯度正常）
    scaler.update()                        # 动态调整缩放因子
```

**为什么续训要保存 scaler？**

如果不保存 scaler → 缩放因子归零或默认值 → 训练不稳定时的"保守策略"丢失 → 可能再次触发梯度爆炸。保存 scaler → 缩放因子恢复到中断前的状态 → 训练平滑继续。

> 大白话：GradScaler 就像一个"自动放大镜"。FP16 的精度太低，小梯度会被"看不清"（下溢为 0）。GradScaler 先把梯度放大（用放大镜看），算完再缩小回去（还原真实尺度）。如果放大镜倍数太大导致"眩光"（inf/nan），就自动降低倍数。续训时必须把这个"放大镜的当前倍数"也存起来，否则又要从头摸索。
