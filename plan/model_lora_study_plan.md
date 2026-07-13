# model_lora.py 学习计划（LoRA 底层实现）

## 一、写在前面：这个文件是干什么的？

你已经学过 `train_lora.py`（LoRA 训练脚本），知道 LoRA 的训练流程。但那个脚本里只调用了 `apply_lora(model)` 这一行就"魔法般"地让模型具备了 LoRA 能力。

**`model_lora.py` 就是这个"魔法"的实现者。** 整个文件只有 150 行，做了四件事：

```
model_lora.py 的四件事：
┌─────────────────────────────────────────────────┐
│ 1. LoRA 类      → 定义"旁路"长什么样（A 降维 → B 升维）  │
│ 2. apply_lora   → 把旁路"嫁接"到模型的 Linear 层上     │
│ 3. load_lora    → 把训练好的旁路权重读回来              │
│ 4. save_lora    → 把训练好的旁路权重存出去              │
└─────────────────────────────────────────────────┘
```

### 学完这篇你能回答的问题

| 问题 | 涉及的函数 |
|------|-----------|
| LoRA 的 A 和 B 矩阵为什么要用不同的初始化方式？ | `LoRA.__init__` |
| `apply_lora` 是怎么"偷天换日"替换 forward 的？ | `apply_lora` |
| 为什么只对 `nn.Linear` 且矩阵是方阵的层加 LoRA？ | `apply_lora` 的 if 条件 |
| `forward_with_lora` 里的闭包变量捕获是怎么回事？ | `forward_with_lora` |
| `load_lora` 是怎么从一个大字典里精准提取 LoRA 权重的？ | `load_lora` |
| `save_lora` 为什么不保存原模型的参数？ | `save_lora` |
| 为什么需要 `_orig_mod` 这个兼容处理？ | `save_lora` |

### 阅读姿势

1. **先看 LoRA 类**（第二章），理解旁路的数学结构
2. **再看 apply_lora**（第三章），理解注入机制——这是最核心的部分
3. **最后看 load/save**（第四章），理解权重的持久化
4. **做自测**（末尾 Q&A），检验是否真的理解

---

## 二、LoRA 类：旁路的数学结构

**位置**：第 7-57 行

### 2.1 结构定义

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank

        self.A = nn.Linear(in_features, rank, bias=False)    # 降维：d → r
        self.B = nn.Linear(rank, out_features, bias=False)    # 升维：r → d

        self.A.weight.data.normal_(mean=0.0, std=0.02)        # A：随机初始化
        self.B.weight.data.zero_()                             # B：全 0 初始化
```

### 2.2 数据流图

```
输入 x: [batch, seq_len, 512]
              │
              ▼
         ┌─────────┐
         │ A 矩阵  │  nn.Linear(512, rank)   降维
         └────┬────┘
              │
              ▼
    中间态: [batch, seq_len, rank]    例如 rank=8 → 从 512 压到 8
              │
              ▼
         ┌─────────┐
         │ B 矩阵  │  nn.Linear(rank, 512)   升维
         └────┬────┘
              │
              ▼
    输出: [batch, seq_len, 512]       恢复到原始维度
```

> 大白话：想象你在压缩图片。A 矩阵把一张 512×512 的大图压缩成 8×8 的缩略图（保留最关键的信息），B 矩阵再把 8×8 的缩略图还原成 512×512。还原出来的图和原图有差异，这个差异就是 LoRA 学到的"增量知识"。

### 2.3 为什么 A 随机初始化、B 全 0 初始化？

这是 LoRA 最精妙的设计之一：

**初始状态的数学保证**：

```python
# B 初始化为全 0
# A 初始化为随机值（比如 [[0.01, -0.02, ...], ...]）

# 前向传播：
output = B(A(x))

# A(x) = 某个随机向量（非零）
# B(某个向量) = 0（因为 B 的权重全是 0）
# 所以 output = 0
```

**训练开始时**：

```
模型总输出 = W₀(x) + B(A(x))
           = W₀(x) + 0          ← LoRA 贡献为零
           = W₀(x)              ← 和没加 LoRA 时完全一样！
```

这意味着：**插入 LoRA 的瞬间，模型的行为不会发生任何变化**。训练从原模型的"完美状态"开始，LoRA 只是慢慢学一个微小的增量。

**如果 B 也随机初始化会怎样？**

```
B 随机 → B(A(x)) ≠ 0 → 模型总输出 = W₀(x) + 随机噪声
                        → 模型瞬间"变傻" → 训练从一个很烂的状态开始
```

> 大白话：B 全 0 就像给模型装了一个"静音开关"——刚装上去时 LoRA 不发出任何声音，模型和以前一模一样。训练开始后，B 的值慢慢从 0 变成非零，LoRA 的声音逐渐出现，模型在原模型的基础上越来越"聪明"。

### 2.4 参数量对比

以 MiniMind Small（hidden_size=512）为例：

```
假设 LoRA 应用在 q_proj（Linear(512, 512)）上，rank=8：

原始 q_proj:  512 × 512 = 262,144 参数（冻结，不训练）
LoRA A:       512 × 8   =   4,096 参数（可训练）
LoRA B:       8 × 512   =   4,096 参数（可训练）
LoRA 总计:                  8,192 参数

占比: 8,192 / 262,144 ≈ 3.1%  ← 仅用 3% 的参数就能适配新任务！
```

---

## 三、apply_lora：把旁路"嫁接"到模型上

**位置**：第 63-77 行

这是整个文件**最核心、也最难理解**的部分。它用了一个 Python 技巧——**闭包 + 函数替换**——在运行时"偷天换日"地修改模型行为。

### 3.1 逐行拆解

```python
def apply_lora(model, rank=8):
    # ① 遍历模型中所有子模块
    for name, module in model.named_modules():

        # ② 筛选条件：必须是 nn.Linear 且是方阵
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:

            # ③ 创建 LoRA 模块
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(module.weight.device)

            # ④ 把 LoRA 挂到模块上
            setattr(module, "lora", lora)
            setattr(module, "lora_list", [lora])

            # ⑤ 保存原始 forward
            original_forward = module.forward

            # ⑥ 定义新的 forward（闭包）
            def forward_with_lora(x, layer1=original_forward, lora_modules=module.lora_list):
                lora_out = sum(lm(x) for lm in lora_modules)
                return layer1(x) + lora_out

            # ⑦ 替换 forward
            module.forward = forward_with_lora
```

### 3.2 筛选条件：为什么只对"方阵"加 LoRA？

```python
isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]
```

MiniMind 中所有 `nn.Linear` 层及其形状（Small 配置）：

| 层 | 形状 | 方阵？ | 加 LoRA？ |
|---|---|:---:|:---:|
| q_proj | [512, 512] | ✅ | ✅ |
| k_proj | [128, 512] | ❌ | ❌ |
| v_proj | [128, 512] | ❌ | ❌ |
| o_proj | [512, 512] | ✅ | ✅ |
| gate_proj | [1365, 512] | ❌ | ❌ |
| up_proj | [1365, 512] | ❌ | ❌ |
| down_proj | [512, 1365] | ❌ | ❌ |
| lm_head | [6400, 512] | ❌ | ❌ |

**只有 q_proj 和 o_proj 被加了 LoRA！**

为什么用"方阵"这个条件？

1. **q_proj 和 o_proj 是 Attention 的"入口"和"出口"**，对模型行为影响最大
2. **k_proj 和 v_proj 因为 GQA（分组查询注意力），维度是 128×512，不是方阵**，被过滤掉了
3. **FFN 层的 gate/up/down 维度是 512→1365→512，都不是方阵**，也被过滤掉了
4. **lm_head 是 6400×512**，维度差异太大，不适合加 LoRA

> 大白话：这个筛选条件就像一个"门卫"，只让"正方形的房间"进去。Attention 的 q_proj 和 o_proj 刚好是正方形（512×512），所以被选中。其他层要么太扁（128×512），要么太长（512×1365），都被挡在门外了。

### 3.3 闭包技巧：`forward_with_lora` 是怎么工作的？

```python
original_forward = module.forward    # 保存原始 forward

def forward_with_lora(x, layer1=original_forward, lora_modules=module.lora_list):
    lora_out = sum(lm(x) for lm in lora_modules)
    return layer1(x) + lora_out

module.forward = forward_with_lora   # 替换 forward
```

这里有一个 Python 闭包的经典用法：

```python
# 假设 module 是 q_proj，它的 forward 原本是：
# q_proj.forward(x) = x @ W_q + b

# 替换后：
# q_proj.forward(x) = x @ W_q + b  +  LoRA(x)
#                      ↑              ↑
#                   原始 forward    LoRA 旁路
```

**关键问题：为什么用默认参数 `layer1=original_forward` 而不是直接引用？**

```python
# ❌ 错误写法（循环变量捕获问题）
for name, module in model.named_modules():
    original_forward = module.forward
    def forward_with_lora(x):
        return original_forward(x) + ...  # original_forward 指向最后一个 module！
    module.forward = forward_with_lora

# ✅ 正确写法（用默认参数捕获）
for name, module in model.named_modules():
    original_forward = module.forward
    def forward_with_lora(x, layer1=original_forward, ...):
        return layer1(x) + ...  # layer1 在定义时就被固定了
    module.forward = forward_with_lora
```

**原因**：Python 的闭包是"延迟绑定"的。如果你在循环里直接引用 `original_forward`，它会指向循环结束后最后一个 module 的 forward。用默认参数 `layer1=original_forward` 可以在定义时"快照"住当前值。

> 大白话：想象你在一个工厂流水线上给每个工位贴标签。如果用"那个工人的名字"（变量引用），最后所有标签都指向最后一个工人。如果用"把名字写在纸上贴上去"（默认参数快照），每个标签就指向正确的工人了。

### 3.4 完整替换流程图

```
apply_lora(model) 被调用后：

模型结构变化（以 q_proj 为例）：

之前：
  q_proj.forward(x) = x @ W_q

之后：
  q_proj.forward(x) = x @ W_q  +  B(A(x))
                       ↑           ↑
                    原始计算      LoRA 旁路
                    （冻结）     （可训练）

其他层（如 k_proj, gate_proj 等）没有被修改，forward 保持不变。
```

#### Q: apply_lora 会替换哪些层？k_proj 和 v_proj 会被替换吗？

**不会替换 k_proj 和 v_proj**。筛选条件 `shape[0] == shape[1]`（方阵）在 MiniMind Small 配置下的实际结果：

| 层 | 形状 | 方阵？ | 注入 LoRA？ |
|---|---|:---:|:---:|
| **q_proj** | [512, 512] | ✅ | ✅ |
| **k_proj** | [128, 512] | ❌ | ❌ |
| **v_proj** | [128, 512] | ❌ | ❌ |
| **o_proj** | [512, 512] | ✅ | ✅ |
| gate_proj | [1408, 512] | ❌ | ❌ |
| up_proj | [1408, 512] | ❌ | ❌ |
| down_proj | [512, 1408] | ❌ | ❌ |
| lm_head | [6400, 512] | ❌ | ❌ |

只有 **q_proj 和 o_proj** 被注入 LoRA，其他层全部跳过。原因是：
- k_proj、v_proj 因为 GQA 缩小了输出维度（`2×64=128`），不是方阵
- FFN 的 gate/up/down 要膨胀到中间维度（1408），不是方阵
- lm_head 要映射到词表大小（6400），不是方阵

这是一个**很保守的策略**——只对 Attention 的"入口"和"出口"加 LoRA，其他层不动。

> 大白话：`apply_lora` 的"门卫"只放行正方形的房间（512×512）。k_proj 和 v_proj 因为 GQA 缩小了（128×512），FFN 层因为要膨胀到中间维度（1408×512），都不是正方形，全被挡在门外了。最终只有 q_proj（入口）和 o_proj（出口）被加了 LoRA。

#### Q: o_proj 是什么？在 Attention 中起什么作用？

**o_proj = Output Projection（输出投影层）**，是 Attention 模块的"最后一关"。它把多头注意力的拼接输出映射回模型的隐藏层维度。

Attention 内部的数据流：

```
输入 x: [batch, seq, 512]
         │
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
  q_proj    k_proj     v_proj        ← 输入投影（产生 Q, K, V）
 [512→512] [512→128] [512→128]
    │         │          │
    ▼         ▼          ▼
  Q [8头]   K [2头]    V [2头]       ← 重塑为多头格式
    │         │          │
    │    ┌────┘          │
    │    │  repeat_kv    │
    │    │  K→8头,V→8头  │
    │    ▼               │
    │  Attention 计算     │           ← Q×K^T → softmax → scores×V
    │    │               │
    │    ▼               │
    │  output [batch, seq, 8, 64]    ← 8 个头的输出
    │    │
    │    ▼ reshape
    │  [batch, seq, 512]             ← 8×64=512，拼接回来
    │    │
    │    ▼
    │  ┌──────────────┐
    │  │   o_proj     │              ← 输出投影！融合 8 个头的信息
    │  │  512 → 512   │                 （被 LoRA 注入的地方）
    │  └──────┬───────┘
    │         │
    └────────▶│
              ▼
        output: [batch, seq, 512]
```

**为什么需要 o_proj？**

多头注意力的输出虽然维度刚好是 512（8×64），但这个 512 是"8 个头各自独立计算后拼起来的"，不同头的信息是割裂的。o_proj 做一次线性变换，让这 512 维的信息重新融合——相当于让 8 个分析师把各自的报告合并成一份综合报告。

**为什么 o_proj 是方阵？**

```
o_proj 输入:  num_heads × head_dim = 8 × 64 = 512
o_proj 输出:  hidden_size = 512
→ nn.Linear(512, 512)  ← 方阵！所以被 apply_lora 注入 LoRA
```

这是模型设计的结果——当 `num_heads × head_dim = hidden_size` 时，o_proj 天然是方阵。

> 大白话：Attention 就像 8 个分析师各自独立分析同一段文字，每人输出 64 维的报告。o_proj 就是"主编"——把 8 份报告合并成一份 512 维的综合报告，确保不同分析师的发现能互相补充。o_proj 之所以被 LoRA 注入，是因为它的维度刚好是"正方形"（512×512）。

### 3.5 apply_lora_multi：多 LoRA 合并

```python
def apply_lora_multi(model, ranks=None):
    if ranks is None:
        ranks = [8]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora_modules = nn.ModuleList()
            for r in ranks:
                lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=r)
                lora_modules.append(lora)
            setattr(module, "lora_list", lora_modules)

            original_forward = module.forward
            def forward_with_lora(x, layer1=original_forward, lms=lora_modules):
                lora_out = sum(lm(x) for lm in lms)
                return layer1(x) + lora_out
            module.forward = forward_with_lora
```

和 `apply_lora` 的区别：

| | apply_lora | apply_lora_multi |
|---|---|---|
| LoRA 数量 | 每层 1 个 | 每层 N 个（N = len(ranks)） |
| lora_list | `[lora]` | `[lora_r8, lora_r16, ...]` |
| forward | `W₀(x) + lora₁(x)` | `W₀(x) + lora₁(x) + lora₂(x) + ...` |
| 用途 | 单 LoRA 训练 | 多 LoRA 合并推理（如不同任务的 LoRA 加权混合） |

---

## 四、load_lora / save_lora：权重的持久化

### 4.1 save_lora：只存 LoRA，不存原模型

**位置**：第 136-150 行

```python
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}

    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)

    torch.save(state_dict, path)
```

**为什么只存 LoRA？**

```
原模型参数:  ~26MB（MiniMind Small）
LoRA 参数:  ~1MB（rank=8 时）

如果存整个模型: 26MB
只存 LoRA:     1MB  ← 节省 96% 的存储空间！
```

**为什么需要 `_orig_mod` 兼容？**

```python
raw_model = getattr(model, '_orig_mod', model)
```

当模型被 `torch.compile()` 编译后，PyTorch 会把原始模型包装成 `OptimizedModule`，原始模型存在 `_orig_mod` 属性里。如果不做这个兼容，`named_modules()` 遍历的是包装后的结构，可能找不到 LoRA 属性。

> 大白话：`torch.compile` 就像给模型穿了一件"加速外套"。`_orig_mod` 就是脱掉外套后里面的那个原始模型。save_lora 需要找到原始模型才能正确提取 LoRA 权重。

### 4.2 load_lora：精准提取 LoRA 权重

**位置**：第 104-112 行

```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cpu')
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora_list') and len(module.lora_list) == 1:
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            if lora_state:
                module.lora_list[0].load_state_dict(lora_state)
```

**权重文件的 key 格式**：

```
保存时的 key 格式:
"model.layers.0.self_attn.q_proj.lora.A.weight"
"model.layers.0.self_attn.q_proj.lora.B.weight"

加载时需要转换为 LoRA 模块的 key:
"A.weight"
"B.weight"
```

**加载流程图**：

```
1. 读取 state_dict（一个大字典）
   {
     "model.layers.0.self_attn.q_proj.lora.A.weight": tensor(...),
     "model.layers.0.self_attn.q_proj.lora.B.weight": tensor(...),
     "model.layers.0.self_attn.o_proj.lora.A.weight": tensor(...),
     ...
   }

2. 遍历模型的每个模块
   找到 module.name = "model.layers.0.self_attn.q_proj"
   且有 lora_list 属性

3. 从 state_dict 中筛选匹配的 key
   筛选条件: key 包含 "model.layers.0.self_attn.q_proj.lora."
   转换: 去掉前缀 → "A.weight", "B.weight"

4. 调用 load_state_dict 加载
   lora.A.weight ← state_dict["A.weight"]
   lora.B.weight ← state_dict["B.weight"]
```

### 4.3 load_lora_multi：多 LoRA 加载

**位置**：第 115-130 行

```python
def load_lora_multi(model, paths, merge_weights=None):
    if merge_weights is None:
        merge_weights = [1.0] * len(paths)
    for idx, path in enumerate(paths):
        state_dict = torch.load(path, ...)
        # ... 省略 DDP 兼容 ...
        for name, module in model.named_modules():
            if hasattr(module, 'lora_list') and idx < len(module.lora_list):
                lora_key_prefix = f'{name}.lora_list.{idx}.'
                lora_state = {k.replace(lora_key_prefix, ''): v for k, v in state_dict.items() if lora_key_prefix in k}
                if lora_state:
                    module.lora_list[idx].load_state_dict(lora_state)
                    if merge_weights[idx] != 1.0:
                        for p in module.lora_list[idx].parameters():
                            p.data.mul_(merge_weights[idx])
```

**merge_weights 参数的作用**：

```
假设有两个 LoRA：
  LoRA_1: 学会了"数学能力"
  LoRA_2: 学会了"编程能力"

merge_weights=[0.7, 0.3] 表示：
  最终输出 = W₀(x) + 0.7 × LoRA_1(x) + 0.3 × LoRA_2(x)
                                   ↑               ↑
                            数学权重 70%     编程权重 30%
```

> 大白话：merge_weights 就像调音台上的音量推子。LoRA_1 是"数学频道"，LoRA_2 是"编程频道"，你可以通过推子控制每个频道的音量，最后混音输出。

---

## 五、完整生命周期：从训练到推理

```
训练阶段：
┌─────────────────────────────────────────────────────────┐
│ 1. apply_lora(model)           给模型加上 LoRA 旁路        │
│ 2. 冻结原模型参数               model.parameters() 不传给优化器 │
│ 3. 训练循环                     只更新 LoRA 的 A/B 权重      │
│ 4. save_lora(model, path)      只保存 LoRA 权重（~1MB）     │
└─────────────────────────────────────────────────────────┘

推理阶段：
┌─────────────────────────────────────────────────────────┐
│ 1. 加载原模型                   model = MiniMindForCausalLM() │
│ 2. apply_lora(model)           给模型加上 LoRA 旁路（B 初始化为 0）│
│ 3. load_lora(model, path)      把训练好的 LoRA 权重加载进来    │
│ 4. 推理                        model(x) = W₀(x) + B(A(x))  │
└─────────────────────────────────────────────────────────┘

合并推理阶段（可选）：
┌─────────────────────────────────────────────────────────┐
│ 1. apply_lora_multi(model, ranks=[8, 16])               │
│ 2. load_lora_multi(model, [path1, path2], [0.7, 0.3])   │
│ 3. 推理: model(x) = W₀(x) + 0.7×LoRA₁(x) + 0.3×LoRA₂(x)│
└─────────────────────────────────────────────────────────┘
```

---

## 六、代码结构总览

```
model_lora.py（150 行）
│
├── LoRA 类（L7-57）              LoRA 旁路的定义（A 降维 + B 升维）
│
├── apply_lora（L63-77）          单 LoRA 注入（闭包替换 forward）
├── apply_lora_multi（L80-98）    多 LoRA 注入（支持多 rank 合并）
│
├── load_lora（L104-112）         单 LoRA 权重加载
├── load_lora_multi（L115-130）   多 LoRA 权重加载（支持加权合并）
│
└── save_lora（L136-150）         LoRA 权重保存（只存 LoRA，不存原模型）
```

### 与 model_minimind.py 的关系

```
model_minimind.py
  ├── Attention.q_proj  →  nn.Linear(512, 512)  →  apply_lora 会注入
  ├── Attention.k_proj  →  nn.Linear(128, 512)  →  不是方阵，跳过
  ├── Attention.v_proj  →  nn.Linear(128, 512)  →  不是方阵，跳过
  ├── Attention.o_proj  →  nn.Linear(512, 512)  →  apply_lora 会注入
  ├── FFN.gate_proj     →  nn.Linear(1365, 512) →  不是方阵，跳过
  ├── FFN.up_proj       →  nn.Linear(1365, 512) →  不是方阵，跳过
  ├── FFN.down_proj     →  nn.Linear(512, 1365) →  不是方阵，跳过
  └── lm_head           →  nn.Linear(6400, 512) →  不是方阵，跳过
```

---

## 七、检查你是否真的理解（Q&A）

### 基础

**1. LoRA 的 A 和 B 矩阵为什么要用不同的初始化方式？**

答案：B 全 0 初始化保证训练开始时 `B(A(x)) = 0`，即 LoRA 的输出为零，模型行为和没加 LoRA 时完全一致。这样训练从原模型的"完美状态"开始，LoRA 只学增量。如果 B 也随机初始化，`B(A(x)) ≠ 0`，模型瞬间被破坏，训练从一个很烂的状态开始。A 用随机初始化是为了打破对称性，让梯度能正确更新。

**2. `apply_lora` 的筛选条件 `module.weight.shape[0] == module.weight.shape[1]` 是什么意思？为什么要加这个条件？**

答案：这个条件筛选"方阵"（行数 = 列数的矩阵）。在 MiniMind 中，只有 `q_proj`（512×512）和 `o_proj`（512×512）满足这个条件。加这个条件的原因：(1) q_proj 和 o_proj 是 Attention 的"入口"和"出口"，对模型行为影响最大；(2) k_proj/v_proj 因为 GQA 维度是 128×512（非方阵）；(3) FFN 层维度是 512→1365→512（非方阵）；(4) lm_head 是 512→6400（非方阵）。

**3. `forward_with_lora` 为什么用默认参数 `layer1=original_forward` 而不是直接引用变量？**

答案：Python 闭包是"延迟绑定"的。如果直接引用 `original_forward` 变量，所有替换后的 forward 都会指向循环结束后最后一个 module 的 forward（因为变量被覆盖了）。用默认参数 `layer1=original_forward` 可以在定义时"快照"住当前值，确保每个模块的 forward 闭包捕获的是正确的 `original_forward`。

### 深入

**4. `load_lora` 中 `state_dict = {(k[7:] if k.startswith('module.') else k): v ...}` 这行在做什么？**

答案：处理 DDP（分布式数据并行）的兼容性。DDP 包装模型时会在每个 key 前面加 `"module."` 前缀（例如 `"module.model.layers.0...."`）。这行代码把 `"module."` 前缀去掉，让 key 格式和模型中的命名一致，确保后续匹配能正确进行。

**5. `save_lora` 中 `getattr(model, '_orig_mod', model)` 在做什么？**

答案：处理 `torch.compile()` 的兼容性。`torch.compile()` 会把原始模型包装成 `OptimizedModule`，原始模型存在 `_orig_mod` 属性里。`getattr` 会优先返回 `_orig_mod`（即原始模型），如果不存在就返回 `model` 本身。这样无论模型是否被 compile 过，都能正确遍历 LoRA 模块。

**6. `load_lora_multi` 中的 `merge_weights` 参数是怎么实现加权合并的？**

答案：`merge_weights` 是一个权重列表，每个元素对应一个 LoRA 的权重系数。加载每个 LoRA 后，如果权重不是 1.0，就用 `p.data.mul_(merge_weights[idx])` 把该 LoRA 的所有参数乘以权重系数。最终效果是 `output = W₀(x) + w₁×LoRA₁(x) + w₂×LoRA₂(x) + ...`，实现了多个 LoRA 的加权混合。

**7. 为什么 LoRA 只保存 A/B 权重，不保存原模型的参数？**

答案：原模型参数是冻结的，训练过程中没有变化，保存了也没用。LoRA 参数才是"学到的新知识"，是每个任务独有的。只保存 LoRA 可以：(1) 极大节省存储（1MB vs 26MB）；(2) 方便多任务切换（加载不同 LoRA 文件即可）；(3) 方便分享（只需要传递小文件）。

---

## 八、动手练习

### 基础

**练习 1：打印 LoRA 注入前后的模型结构**

修改 `train_lora.py`，在 `apply_lora(model)` 前后各打印一次 `model.named_modules()`，观察哪些层被加了 LoRA 属性。

**练习 2：验证 LoRA 初始输出为零**

```python
# 在 apply_lora 之后，手动验证
x = torch.randn(1, 10, 512)
for name, module in model.named_modules():
    if hasattr(module, 'lora'):
        lora_out = module.lora(x)
        print(f"{name} LoRA output max: {lora_out.abs().max().item()}")
        # 应该接近 0（因为 B 全 0 初始化）
```

### 进阶

**练习 3：统计 LoRA 注入的层数和参数量**

```python
total_lora_params = 0
lora_count = 0
for name, module in model.named_modules():
    if hasattr(module, 'lora'):
        lora_count += 1
        total_lora_params += sum(p.numel() for p in module.lora.parameters())
print(f"共注入 {lora_count} 层 LoRA")
print(f"LoRA 总参数量: {total_lora_params:,}")
print(f"占原模型参数量的比例: {total_lora_params / sum(p.numel() for p in model.parameters()) * 100:.2f}%")
```

**练习 4：理解闭包变量捕获**

写一个简单的 Python 脚本，对比"直接引用变量"和"默认参数快照"在循环中的行为差异，验证闭包的延迟绑定问题。

### 深入

**练习 5：对比 apply_lora 和 apply_lora_multi 的 forward 差异**

修改代码，在两个函数的 `forward_with_lora` 中加入 print 语句，观察单 LoRA 和多 LoRA 的调用路径有何不同。

**练习 6：手动实现一个简版 save_lora**

不看源码，自己实现一个 `save_lora` 函数，只提取 LoRA 的 A/B 权重并保存。然后对比你的实现和源码的差异。
