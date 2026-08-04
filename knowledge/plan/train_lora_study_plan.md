# train_lora.py 学习计划指引

## 一、文件定位

`train_lora.py` 是 MiniMind 项目使用 **LoRA（Low-Rank Adaptation）** 进行参数高效微调的脚本。
与 `train_full_sft.py` 几乎完全一样，但多了一个关键步骤：**冻结原模型权重，只训练低秩矩阵**。

```
train_pretrain.py（预训练，✅ 已完成）
    ↓ 产出 pretrain.pth
train_full_sft.py（指令微调，✅ 已完成）
    ↓ 产出 full_sft.pth
train_lora.py（LoRA 参数高效微调，← 你在这里）
    ↓ 产出 lora_xxx.pth（极小，仅 ~1MB）
train_dpo.py → train_reason.py → train_grpo.py（后续阶段）
```

### 为什么需要 LoRA？

Full SFT 虽然效果好，但**每个任务都要保存一份完整模型权重**（~300MB 以上）。
LoRA 可以在冻结原模型的情况下，用**极少的额外参数**（~1MB）适配特定任务：

| | Full SFT | LoRA |
|--|---------|------|
| 可训练参数量 | 100%（全参） | ~0.1%~1%（仅低秩矩阵） |
| 每任务权重大小 | ~300MB+ | ~1MB |
| 部署切换成本 | 加载完整新模型 | 加载基础模型 + 切换 LoRA 文件 |
| 训练显存 | 高（需存全参数梯度） | 低（仅 LoRA 参数有梯度） |
| 多任务支持 | N 个任务 → N 份完整权重 | N 个任务 → N 份小 LoRA 权重 |
| 效果 | 上限最高 | 接近 full SFT（rank 足够时） |

## 二、核心概念

### 2.1 LoRA 的数学原理

核心公式：
```
h = W₀x + ΔWx = W₀x + BAx
```

其中：
- `W₀ ∈ ℝ^{d×d}`：原始权重矩阵（**冻结**，不训练）
- `B ∈ ℝ^{d×r}`：升维矩阵（可训练）
- `A ∈ ℝ^{r×d}`：降维矩阵（可训练）
- `r ≪ d`：秩（rank），通常是 8、16、32
- `ΔW = BA`：低秩更新矩阵（秩 ≤ r，远小于 d）

**为什么有效**：
预训练模型的内在维度（intrinsic dimension）很低——虽然权重的参数空间很大，
但有效的变化实际上被约束在一个低维子空间中。LoRA 显式地在这个低维子空间中搜索更新方向。

### 2.2 初始化策略

```
A: normal(0, 0.02)    ← 高斯初始化，打破对称性
B: zero               ← 全 0 初始化，保证 BAx = 0
                       → 插入 LoRA 后模型输出不变
                       → 不影响已有知识，从零开始学习新任务
```

### 2.3 旁路结构

```
    输入 x
      │
      ├────→ W₀ (冻结) ──→ h_base ──┐
      │                               │
      └────→ A → B (可训练) ─→ Δh ──┼──→ h = h_base + Δh
                                      │
                                 最终输出
```

## 三、关键代码路径

### 3.1 LoRA 模块定义（scripts/Model/model_lora.py）

```
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)  # 降维
        self.B = nn.Linear(rank, out_features, bias=False)  # 升维

    def forward(self, x):
        return self.B(self.A(x))  # 先降维再升维
```

### 3.2 模块注入（apply_lora）

遍历模型所有子模块，对每个方阵 Linear 层注入 LoRA：

```
apply_lora(model):
    for each nn.Linear in model:
        if weight.shape[0] == weight.shape[1]:      # 只对方阵注入
            lora = LoRA(in_features, out_features, rank)
            module.lora = lora                       # 保存为模块属性
            module.forward = forward_with_lora       # monkey-patch forward
```

`forward_with_lora` 的实现：
```python
def forward_with_lora(x):
    return original_forward(x) + lora(x)  # 原输出 + LoRA 旁路输出
```

### 3.3 参数冻结（train_lora.py:132-139）

```python
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True       # LoRA 参数可训练
        lora_params.append(param)
    else:
        param.requires_grad = False      # 原模型参数冻结
```

**关键差异**：
- Full SFT：`optimizer = AdamW(model.parameters(), ...)`
- LoRA SFT：`optimizer = AdamW(lora_params, ...)`
- 梯度裁剪：`clip_grad_norm_(lora_params, ...)` 而不是 `model.parameters()`

### 3.4 LoRA 权重保存（save_lora）

只保存 LoRA 参数，不保存完整模型：
```python
def save_lora(model, path):
    for each module:
        if hasattr(module, 'lora'):
            state_dict.update(lora.state_dict())
    torch.save(state_dict, path)  # 极小文件（~1MB）
```

### 3.5 多 LoRA 合并（apply_lora_multi / load_lora_multi）

单个 Linear 层可以有多个 LoRA 模块（lora_list），推理时选择性地加载：
```python
forward_with_lora(x):
    lora_out = sum(lm(x) for lm in lora_modules)  # 多个 LoRA 输出叠加
    return original_forward(x) + lora_out
```

支持不同权重比例合并（merge_weights），实现多任务 LoRA 插值。

### 3.6 命令行参数差异（和 full SFT 相比）

| 参数 | full SFT | LoRA |
|------|---------|------|
| `save_weight` / `lora_name` | `full_sft` | `lora_identity` |
| `--epochs` | 2 | 50（LoRA 收敛需要更多步数） |
| `--batch_size` | 16 | 32（显存占用小，可以更大） |
| `--learning_rate` | 1e-6 | 1e-4（LoRA 可以更大） |
| `--from_weight` | `pretrain` | `full_sft`（基于 SFT 权重） |
| `save_dir` | `../models` | `../models/lora` |
| 数据集 | `sft_mini_512.jsonl` | `lora_identity.jsonl`（通常是单任务小数据集） |

### 3.7 LoRA 权重加载（推理）

`eval_llm.py` 中的 LoRA 加载逻辑：
```python
if lora_path:
    apply_lora(model)               # 重新注入 LoRA 结构
    load_lora(model, lora_path)     # 加载训练好的 LoRA 权重
```

**注意**：加载 LoRA 时必须先调用 `apply_lora` 再 `load_lora`，
因为 `load_lora` 依赖每个 Linear 模块上的 `lora` 属性。

## 四、学习目标检查清单（含详解）

### □ 理解 LoRA 的核心思想（低秩分解，冻结原权重）

LoRA 的核心洞察：微调产生的权重变化 ΔW 是低秩的（信息集中在少量方向上）。
因此不直接学 ΔW (d×k)，而是分解为 B(d×r) × A(r×k)。冻结原始 W₀ 不变，
只训练极小 B 和 A，用 r(d+k) 参数模拟 d×k 的更新，省 97%+ 参数量。

### □ 理解 LoRA 前向传播流程（旁路加法）

```
输入 x → W₀(x) = h_base  ──┐
       → B(A(x)) = Δh     ──┼── 输出 = h_base + Δh
```

BA 是**旁路分支**，不修改主路径 W₀。前向时主路 + 旁路输出逐元素相加。
推理时可把 BA「合并」回 W₀（W₀' = W₀ + BA），恢复为单路，无推理延迟。

### □ 理解 apply_lora 的模块注入机制

遍历模型的 `named_modules()`，对每个**方阵 Linear 层**（shape[0]==shape[1]）：
1. 创建 LoRA 实例，挂为 `module.lora` 属性
2. 用闭包 `forward_with_lora` 替换 `module.forward`
3. `forward_with_lora` 内调用 `original_forward(x) + lora(x)`（旁路加法）

这本质是 monkey-patch：不改模型定义，运行时替换目标函数。

### □ 理解参数冻结与仅训练 LoRA 权重的梯度控制

```python
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True    # 仅 LoRA 参数可训
        lora_params.append(param)
    else:
        param.requires_grad = False   # 冻结原模型全部参数
```

效果：反向传播只为 LoRA 参数计算梯度，原模型参数 `.grad = None`，
优化器状态量极小（只保存 LoRA 参数的 momentum / variance）。

### □ 理解梯度裁剪作用在 lora_params 而非 model.parameters()

clip_grad_norm_ 需要传入**有梯度**的参数列表。
原模型参数 requires_grad=False，梯度为 None，传给 clip_grad_norm_ 会报错。
lora_params 恰是全部有梯度的参数集合，所以裁剪 lora_params = 裁剪全模型中会更新的部分。

### □ 理解 LoRA 权重的保存/加载机制（save_lora / load_lora）

- **save_lora**：遍历模型，只提取 `module.lora` 的 state_dict，保存为极小 .pth 文件（~1MB）
- **load_lora**：先 `apply_lora` 重建 LoRA 结构，再从 .pth 恢复 A/B 权重
- 推理时**必须**按 `apply_lora → load_lora` 顺序执行，因为 load 依赖 `module.lora` 属性存在

### □ 对比 full SFT 和 LoRA 的异同（参数量、显存、效果）

| 维度 | Full SFT | LoRA SFT |
|------|---------|----------|
| 可训练参数 | 100% | ~0.1%~1%（仅 BA） |
| 每任务存储 | ~300MB+（完整权重） | ~1MB（仅 LoRA 权重） |
| 训练显存 | 高（全参梯度+优化器状态） | 低（仅 LoRA 有梯度） |
| 学习率 | 1e-6（微调已有知识） | 1e-4（从零学新分支） |
| epoch | 2（全参收敛快） | 50（少量参数需更多步） |
| 效果上限 | 最高（全参更新） | 接近 full SFT（rank 足够时） |
| 多任务 | N 份完整权重 | 1 份基座 + N 份 LoRA 文件 |

### □ 理解多 LoRA 合并与权重融合（apply_lora_multi / load_lora_multi）← 详细讲解

#### 为什么需要多 LoRA？

实际场景中经常有多个 LoRA 权重（如 `lora_identity.pth`、`lora_medical.pth`、
`lora_law.pth`），每个适配一种任务。如果逐个加载推理，就得 N 次部署 / N 次模型拷贝。
多 LoRA 合并允许**一次性加载多个 LoRA 权重到同一个模型**，前向时按权重比例求和。

#### 数据结构：从单个 lora 到 lora_list

```python
# apply_lora（单 LoRA）：
module.lora = LoRA(...)           # 一个目标层挂一个 LoRA
module.lora_list = [lora]         # 列表里只有一个元素

# apply_lora_multi（多 LoRA）：
module.lora_list = nn.ModuleList(
    [LoRA(..., rank=r1),           # 列表里有多个 LoRA
     LoRA(..., rank=r2),
     LoRA(..., rank=r3)]
)
```

关键区别：`lora_list` 是 `nn.ModuleList`，里面的参数会被 `model.parameters()` 自动纳入，
从而参与训练/保存。但多 LoRA 通常是先训好多个单 LoRA 文件，推理时再合并加载——
所以 `apply_lora_multi` 更多是**推理时**的结构准备。

#### 合并前向：多个旁路叠加

```python
def forward_with_lora(x, layer1=original_forward, lms=lora_modules):
    lora_out = sum(lm(x) for lm in lms)  # 所有 LoRA 旁路输出求和
    return layer1(x) + lora_out           # 再加到原始输出上
```

相当于：`output = W₀x + B₁A₁x + B₂A₂x + B₃A₃x + ...`

每个 LoRA 学习不同的任务适配方向，求和后实现多任务效果融合。

#### load_lora_multi：按权重比例融合

```python
def load_lora_multi(model, paths, merge_weights=None):
    # paths = ["lora_identity.pth", "lora_medical.pth", "lora_law.pth"]
    # merge_weights = [0.5, 0.3, 0.2]  ← 每个 LoRA 的贡献比例

    for idx, path in enumerate(paths):
        state_dict = torch.load(path)
        for name, module in model.named_modules():
            if hasattr(module, 'lora_list') and idx < len(module.lora_list):
                # 提取当前 LoRA 对应位置的权重
                lora_key_prefix = f'{name}.lora_list.{idx}.'
                lora_state = 提取匹配的键值对
                module.lora_list[idx].load_state_dict(lora_state)

                # 按比例缩放权重（权重融合的核心）
                if merge_weights[idx] != 1.0:
                    for p in module.lora_list[idx].parameters():
                        p.data.mul_(merge_weights[idx])
```

**权重融合的关键**：加载每个 LoRA 权重后，立即乘以 `merge_weights[idx]`。
这等价于：`output = W₀x + w₁·B₁A₁x + w₂·B₂A₂x + w₃·B₃A₃x`

- `merge_weights` 之和可以不等于 1（可独立控制每个 LoRA 的强度）
- 权重缩放是在加载时做的，推理时不再有任何额外开销
- **merge_weights 从哪来？** 它是 `load_lora_multi` 的参数，由用户**手动指定**，不是训练出来的。典型设定方式：
  - 等权融合：`[1.0, 1.0, 1.0]` — 每个 LoRA 同等贡献
  - 任务插值：`[0.7, 0.3]` — 70% 通用 + 30% 医疗，平滑过渡
  - 消融实验：`[1.0, 0.0]` — 只加载第一个，验证第二个的效果
  - 搜索调优：在小验证集上扫一组权重组合（如 [0.5,0.5]、[0.6,0.4]、[0.7,0.3]...），选最优

#### 与「合并到原始权重」的区别

注意不要把这里说的"多 LoRA 合并"和「LoRA merge into W₀」混淆：

| 操作 | 场景 | 效果 |
|------|------|------|
| **多 LoRA 合并**（本小节） | 推理 | 多个 LoRA 旁路同时生效，output = W₀ + Σ(wᵢ·BᵢAᵢ) |
| **权重融合（merge into W₀）** | 部署优化 | 把 BA 写回 W₀，W₀' = W₀ + BA，恢复为单 Linear 层 |

本脚本的 `apply_lora_multi` + `load_lora_multi` 实现的是前者——多个 LoRA 旁路叠加。

#### 实际应用场景

1. **多任务插值**：在 identity（通用对话）和 medical（医疗问答）之间插值，设 `weights=[0.7, 0.3]` 得到兼顾通用和专业的回答
2. **任务组合**：同一基座模型同时 loading 多个领域的 LoRA，一次推理覆盖多个场景
3. **模型集成**：不同超参训练出的 LoRA 加权集成，类似模型集成的思路，往往能提升稳定性

## 五、文件逐段精读计划

### 第 1 层：scripts/Model/model_lora.py（核心逻辑，先读这个）

- **LoRA 类**（L7-57）：理解降维 → 升维的旁路结构，重点看初始化策略
- **apply_lora**（L63-77）：理解模块注入的 monkey-patch 原理
- **save_lora / load_lora**（L136-150 / L104-112）：理解权重的精简保存格式

### 第 2 层：train_lora.py（训练脚本）

- **参数冻结**（L132-139）：对比 full SFT 的 `model.parameters()` 差异
- **优化器**（L145）：`AdamW(lora_params, ...)` 而不是全参数
- **梯度裁剪**（L42）：`clip_grad_norm_(lora_params, ...)` 的差异点
- **保存**（L59-62）：`save_lora()` 替代 `torch.save(state_dict)`

### 第 3 层：eval_llm.py 中的 LoRA 加载

- 回顾 `eval_llm.py` 中 `apply_lora` + `load_lora` 的加载流程

## 六、自测题

### 基础题
1. LoRA 的 A 和 B 矩阵分别是什么形状？为什么 A 高斯初始化、B 零初始化？

   **答案**：
   - A：`Linear(in_features, rank)` → 权重形状 `[rank, in_features]`（降维）
   - B：`Linear(rank, out_features)` → 权重形状 `[out_features, rank]`（升维）
   - **B 零初始化**：B 全零 → `B(A(x)) = 0` 恒成立（任何向量乘零矩阵都得零），
     output = W₀x + 0 = W₀x。保证插入 LoRA 那一刻模型行为不变，不破坏预训练知识。
   - **A 高斯初始化**：反向传播时 `∂L/∂B = ∂L/∂output · A(x)ᵀ`，
     若 A 全零 → 梯度传不回 → 什么都学不到；
     若 A 全等（如全 1）→ 所有 rank 维度收到相同输入 → 学相同特征 → r=8 跟 r=1 没区别；
     高斯随机初始化让 A 每行指向输入空间的不同随机方向 → 打破对称性，
     各 LoRA 维度有机会分化出不同功能。
2. 为什么 LoRA 的参数量和 rank 成正比？rank=8 和 rank=64 的参数量差多少倍？

   **答案**：
   - 每层 LoRA 参数量 = B 参数量 + A 参数量 = r×out_features + r×in_features = r×(d+k)
   - 当 d=k=hidden_size 固定时，参数量 ∝ r（线性正比）
   - rank=64 ÷ rank=8 = 8 倍 → rank=64 的参数量是 rank=8 的 8 倍
3. forward_with_lora 的函数签名中，为什么用闭包捕获 original_forward 而不是直接引用？

   **答案**：

   先看错误的写法，理解问题出在哪：

   ```python
   for name, module in model.named_modules():
       if isinstance(module, nn.Linear):
           original_forward = module.forward            # 记下当前层的forward
           lora = LoRA(...)
           module.lora = lora

           def forward_with_lora(x):                    # ❌ 直接引用 module
               return original_forward(x) + module.lora(x)
           
           module.forward = forward_with_lora
   ```

   **问题**：`def forward_with_lora(x)` 里的 `module` 不是一个具体的值，
   它只是一个**名字**。Python 在**调用这个函数时**才会去查 `module` 当前指向谁。
   循环走完后，`module` 变量指向的是**最后一个 Linear 层**。
   所以你不管调用哪个层的 LoRA，它查到的 `module` 都是最后一层→用的都是最后一层的 lora。

   打个比方：你让 10 个人分别记住"当前班长是谁"——结果 10 个人都在最后一刻才去看，
   班长已经换了 10 轮了，10 个人记住的都是最后一任班长。这就是"晚绑定"。

   **解法**：用默认参数把当前值"冻结"进函数定义：

   ```python
   def forward_with_lora(x, layer1=original_forward, lora_module=module.lora):
       return layer1(x) + lora_module(x)
   ```

   Python 的**默认参数在函数定义时求值**（而不是调用时），
   所以定义那一刻 `original_forward` 和 `module.lora` 的值就被"拍照"存进了默认参数。
   每个层都拍了一张自己的照片，后面调用时各看各的照片，不会乱。

   **总结一句话**：不用默认参数捕获→所有层共享循环结束后的最后一个 `module` 值；
   用默认参数捕获→每层在定义时就把自己的值存下来了。

### 进阶题
4. LoRA SFT 中 clip_grad_norm_ 作用在 lora_params 上，而 full SFT 作用在 model.parameters() 上。如果把 LoRA 的梯度裁剪改回 model.parameters() 会怎样？（提示：被冻结的参数梯度为 0，不影响数值，但多了一次遍历开销）

   **答案**：不影响最终数值结果。非 LoRA 参数 requires_grad=False → 梯度为 None，
   clip_grad_norm_ 会跳过 None 梯度（新版 PyTorch 行为），所以裁剪效果与只传 lora_params 完全一样。
   但要多遍历 ~99% 的无关参数，浪费少量时间。
5. save_lora 中为什么要 `getattr(model, '_orig_mod', model)` 扒壳？什么情况下 model 会有 _orig_mod 属性？

   **答案**：`torch.compile(model)` 会在模型外套一层编译优化壳，原始模型被存为 `model._orig_mod`。
   `save_lora` 遍历 module 查找 `.lora` 属性——如果带着 compile 壳遍历，可能找不到正确的子模块。
   用 `getattr(model, '_orig_mod', model)` 先尝试扒壳取原始模型，拿不到就用 model 本身，
   保证一定能找到各个 Linear 层上挂载的 LoRA 权重。
6. 为什么 LoRA 的学习率（1e-4）可以比 full SFT（1e-6）大两个数量级？（提示：LoRA 从零开始学，full SFT 在已有知识上微调）

   **答案**：两个原因叠加：
   - LoRA 的 B 矩阵零初始化 → 初始 LoRA 分支输出为 0 → 需大学习率快速产生有效信号
   - LoRA 参数量极少（~1%）→ 同样的更新量对最终输出影响小 → 需要更大步长才能让模型学到东西
   而 Full SFT 在全参数上微调，模型已有很好的预训练权重，学习率大了会破坏已有知识（灾难性遗忘）。

### 深入题
7. apply_lora 中 `isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]` 仅对方阵注入。在 Transformer 中有哪些 Linear 层是方阵？哪些不是？为什么 LoRA 通常只注入了注意力投影矩阵（Q/K/V/O）？

   **答案**：
   - **方阵 Linear 层**（d×d）：
     Q_proj（查询投影）、K_proj（键投影）、V_proj（值投影）、
     O_proj（注意力输出投影，将拼接后的注意力结果映射回 hidden_size）
   - **非方阵 Linear 层**（如 d×4d / 4d×d）：
     gate_proj、up_proj（FFN 升维层，d→4d，不是方阵）
     down_proj（FFN 降维层，4d→d，也不是方阵）
   - O（Output projection）就是你问的——注意力做完后，把拼接好的多头结果
     映射回 hidden_size 的那个矩阵，和 Q/K/V 一样是方阵。
   - **为什么只注 Q/K/V/O**：这些层的输入输出维度相同，ΔW 是方阵，适合低秩分解。
     FFN 的升维降维层不是方阵（如 512→2048），注入 LoRA 的收益较低，通常不优先考虑。
8. 如果有两个 LoRA 权重（如 lora_identity.pth 和 lora_medical.pth），`apply_lora_multi` 和单 LoRA 的 `apply_lora` 在实现上有什么关键区别？为什么需要 lora_list 而不是单个 lora 属性？

   **答案**：
   - 关键区别不在"两个输出加起来"，而在**数据结构**：
     - `apply_lora`：`module.lora = LoRA(...)`，`lora_list = [lora]`
     - `apply_lora_multi`：`module.lora_list = nn.ModuleList([LoRA(r1), LoRA(r2), ...])`
   - 为什么用 `nn.ModuleList`？因为 PyTorch 只自动注册 `nn.ModuleList` 里的子模块参数。
     如果只用一个普通 Python 列表存多个 LoRA，`model.parameters()` 不会包含它们，
     参数无法训练/保存。
   - 前向时：`sum(lm(x) for lm in lora_modules)` 把多个 LoRA 旁路输出叠加。
9. 如果 rank=8，对于 hidden_size=512 的层，LoRA 参数量是完整 Linear 层（512×512=262K）的百分之多少？如果 hidden_size=4096（7B 模型）呢？这个比例说明了什么？

   **答案**：
   - **hidden_size=512**：LoRA = 512×8 + 8×512 = 8,192 参数，占 8,192/262,144 = **3.13%**
   - **hidden_size=4096**：LoRA = 4096×8 + 8×4096 = 65,536 参数，占 65,536/16,777,216 = **0.39%**
   - **这比例说明了什么？** LoRA 参数量随 hidden_size 线性增长（O(d)），
     而完整 Linear 层随 hidden_size 平方增长（O(d²)）。
     模型越大 → LoRA 占比反而越小 → **LoRA 在大模型上的效率优势更突出**。
     这也是 LoRA 在 7B、13B、70B 级模型上广泛使用的原因。

## 七、关联文件

```
train_lora.py
 ├─ scripts/Model/model_lora.py            ← LoRA 模块定义、注入、保存（核心）
 ├─ scripts/Model/model_minimind.py         ← 被注入 LoRA 的目标模型
 ├─ scripts/Dataset/lm_dataset.py           ← SFTDataset（和 full SFT 完全一样）
 ├─ scripts/Trainer/trainer_utils.py        ← 工具函数（和 full SFT 完全一样）
 ├─ scripts/Trainer/train_full_sft.py       ← 对比参考：full SFT 实现
 └─ plan/train_full_sft_study_plan.md ← 之前的学习笔记
```
