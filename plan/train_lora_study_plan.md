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

### 3.1 LoRA 模块定义（model/model_lora.py）

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
| `save_dir` | `../out` | `../out/lora` |
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

## 四、学习目标检查清单

- [ ] 理解 LoRA 的核心思想（低秩分解，冻结原权重）
- [ ] 理解 LoRA 前向传播流程（旁路加法）
- [ ] 理解 apply_lora 的模块注入机制
- [ ] 理解参数冻结与仅训练 LoRA 权重的梯度控制
- [ ] 理解梯度裁剪作用在 lora_params 而非 model.parameters()
- [ ] 理解 LoRA 权重的保存/加载机制（save_lora / load_lora）
- [ ] 对比 full SFT 和 LoRA 的异同（参数量、显存、效果）
- [ ] 理解多 LoRA 合并与权重融合（apply_lora_multi / load_lora_multi）

## 五、文件逐段精读计划

### 第 1 层：model/model_lora.py（核心逻辑，先读这个）

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
2. 为什么 LoRA 的参数量和 rank 成正比？rank=8 和 rank=64 的参数量差多少倍？
3. forward_with_lora 的函数签名中，为什么用闭包捕获 original_forward 而不是直接引用？

### 进阶题
4. LoRA SFT 中 clip_grad_norm_ 作用在 lora_params 上，而 full SFT 作用在 model.parameters() 上。如果把 LoRA 的梯度裁剪改回 model.parameters() 会怎样？（提示：被冻结的参数梯度为 0，不影响数值，但多了一次遍历开销）
5. save_lora 中为什么要 `getattr(model, '_orig_mod', model)` 扒壳？什么情况下 model 会有 _orig_mod 属性？
6. 为什么 LoRA 的学习率（1e-4）可以比 full SFT（1e-6）大两个数量级？（提示：LoRA 从零开始学，full SFT 在已有知识上微调）

### 深入题
7. apply_lora 中 `isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]` 仅对方阵注入。在 Transformer 中有哪些 Linear 层是方阵？哪些不是？为什么 LoRA 通常只注入了注意力投影矩阵（Q/K/V/O）？
8. 如果有两个 LoRA 权重（如 lora_identity.pth 和 lora_medical.pth），`apply_lora_multi` 和单 LoRA 的 `apply_lora` 在实现上有什么关键区别？为什么需要 lora_list 而不是单个 lora 属性？
9. 如果 rank=8，对于 hidden_size=512 的层，LoRA 参数量是完整 Linear 层（512×512=262K）的百分之多少？如果 hidden_size=4096（7B 模型）呢？这个比例说明了什么？

## 七、关联文件

```
train_lora.py
 ├─ model/model_lora.py            ← LoRA 模块定义、注入、保存（核心）
 ├─ model/model_minimind.py         ← 被注入 LoRA 的目标模型
 ├─ dataset/lm_dataset.py           ← SFTDataset（和 full SFT 完全一样）
 ├─ trainer/trainer_utils.py        ← 工具函数（和 full SFT 完全一样）
 ├─ trainer/train_full_sft.py       ← 对比参考：full SFT 实现
 └─ plan/train_full_sft_study_plan.md ← 之前的学习笔记
```
