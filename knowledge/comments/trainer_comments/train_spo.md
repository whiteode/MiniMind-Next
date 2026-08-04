# train_spo.py 注释整理

> 本文档收录 `trainer/train_spo.py` 中被移除的全部注释与 docstring。
> 按原代码顺序分节，每节对应原代码中的一个逻辑块 / 函数。

---

## 类 `AutoAdaptiveValueTracker`

> 对应原代码：`class AutoAdaptiveValueTracker:`（第 27–28 行）

**docstring：**

```text
SPO自适应价值追踪器
```

---

## 函数 `calculate_rewards`

> 对应原代码：`def calculate_rewards():`（第 69–70 行）

**docstring：**

```text
整合所有奖励函数计算总奖励
```

---

## 函数 `spo_train_epoch`

> 对应原代码：`def spo_train_epoch():`（第 131–148 行）

**注释：**

```text
list[str], length B
input_ids: [B, P], attention_mask: [B, P]
DDP 模型需要使用 .module 访问 generate 方法
[B, P+R]
[B, R]
```

---

## 函数 `get_per_token_logps`

> 对应原代码：`def get_per_token_logps():`（第 149–272 行）

**注释：**

```text
[B, R]
[B, R]
list[str], length B
[B]
[B]
Un-normalize baselines to be in the same scale as raw rewards [-3, 3]
[B]
[B]
直接使用 baseline 提供的优势估计，只做裁剪防止梯度爆炸。不再做 batch 内归一化，因为 baseline 已经提供了跨 batch 的稳定基线
[B, R]
[B]
[B, R]
[B, R]
[B, R]
[B, R]
scalar
[B, R]
```

---

## 1. 初始化环境和随机种子

> 对应原代码：`# ========== 1. 初始化环境和随机种子 ==========` 段（第 273–277 行）

**注释：**

```text
========== 1. 初始化环境和随机种子 ==========
```

---

## 2. 配置目录、模型参数、检查ckp

> 对应原代码：`# ========== 2. 配置目录、模型参数、检查ckp ==========` 段（第 278–283 行）

**注释：**

```text
========== 2. 配置目录、模型参数、检查ckp ==========
```

---

## 3. 设置混合精度

> 对应原代码：`# ========== 3. 设置混合精度 ==========` 段（第 284–288 行）

**注释：**

```text
========== 3. 设置混合精度 ==========
```

---

## 4. 配wandb

> 对应原代码：`# ========== 4. 配wandb ==========` 段（第 289–297 行）

**注释：**

```text
========== 4. 配wandb ==========
```

---

## 5. 初始化模型（Policy, Ref, Reward）和Value Tracker、数据

> 对应原代码：`# ========== 5. 初始化模型（Policy, Ref, Reward）和Value Tracker、数据 ==========` 段（第 298–325 行）

**注释：**

```text
========== 5. 初始化模型（Policy, Ref, Reward）和Value Tracker、数据 ==========
Policy模型
Reference模型
Reward模型
Value Tracker
```

---

## 6. 从ckp恢复状态

> 对应原代码：`# ========== 6. 从ckp恢复状态 ==========` 段（第 326–334 行）

**注释：**

```text
========== 6. 从ckp恢复状态 ==========
```

---

## 7. DDP包模型

> 对应原代码：`# ========== 7. DDP包模型 ==========` 段（第 335–339 行）

**注释：**

```text
========== 7. DDP包模型 ==========
```

---

## 8. 开始训练

> 对应原代码：`# ========== 8. 开始训练 ==========` 段（第 340–352 行）

**注释：**

```text
========== 8. 开始训练 ==========
```

---

## 9. 清理分布进程

> 对应原代码：`# ========== 9. 清理分布进程 ==========` 段（第 353–354 行）

**注释：**

```text
========== 9. 清理分布进程 ==========
```
