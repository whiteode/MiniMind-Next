# convert_model.py 学习计划

> **文件位置**: `scripts/Tools/convert_model.py`（77 行）
> **角色**: 在 PyTorch 原生权重（`.pth`）与 HuggingFace Transformers 格式之间互相转换
> **前置知识**: `model_minimind.py`（模型架构）、已完成至少一个训练阶段（有 `.pth` 权重）

---

## 文件全景图

```
convert_model.py
│
├── convert_torch2transformers_minimind()
│   └── .pth → MiniMindForCausalLM → save_pretrained (MiniMind HF 格式)
│
├── convert_torch2transformers_llama()
│   └── .pth → LlamaForCausalLM → save_pretrained (Llama HF 格式，兼容第三方)
│
├── convert_transformers2torch()
│   └── HF 格式 → .pth（反向转换）
│
└── __main__
    ├── 定义 lm_config (hidden_size=512, 8层, 非 MoE)
    ├── 设置 .pth 路径 → ../models/full_sft_512.pth
    ├── 设置输出路径 → ../MiniMind2-Small
    └── 调用 convert_torch2transformers_llama()
```

---

## 第一章：运行方式

### 1.1 前置条件

```bash
# 确认已训练出 .pth 权重文件
ls ../models/
# 应看到类似: pretrain_512.pth  full_sft_512.pth  dpo_512.pth  ...

# 确认有 tokenizer 文件
ls ../scripts/Model/
# 应看到: tokenizer.json  tokenizer_config.json  ...
```

### 1.2 基本运行

```bash
cd /mnt/data_2t_0/Projects/minimind

# 直接运行（默认配置：hidden_size=512, 8层, 非MoE）
python scripts/Tools/convert_model.py
```

默认行为：
- 输入：`../models/full_sft_512.pth`
- 输出：`../MiniMind2-Small/`（LlamaForCausalLM 格式，兼容第三方生态）
- 精度：`float16`

### 1.3 自定义运行

因为目前 `__main__` 中配置是**硬编码**的，修改模型尺寸需要直接改脚本：

```bash
# 例如转换 768 hidden_size 的模型
# 编辑第 72 行: hidden_size=768, num_hidden_layers=16
# 编辑第 74 行: transformers_path = '../MiniMind2-Base'
python scripts/Tools/convert_model.py
```

### 1.4 转换后验证

```bash
# 查看转换后的文件
ls -la ../MiniMind2-Small/
# 应看到: config.json  pytorch_model.bin  tokenizer.json  tokenizer_config.json  ...

# 用 transformers 加载验证
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('../MiniMind2-Small', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('../MiniMind2-Small')
print(f'加载成功，参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
"
```

---

## 第二章：核心概念——两种 HF 格式

### 2.1 MiniMind HF 格式（`convert_torch2transformers_minimind`）

```python
MiniMindConfig.register_for_auto_class()
MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
model = MiniMindForCausalLM(lm_config)
model.load_state_dict(state_dict)
model.save_pretrained(transformers_path)
```

- 模型结构：`MiniMindForCausalLM`（自定义 `PreTrainedModel`）
- 加载方式：`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`
- 需要 `trust_remote_code=True`，因为 HuggingFace 需要下载并执行自定义模型代码
- 注册了 `model_type = "minimind"`，在 `config.json` 中体现为 `"model_type": "minimind"`

### 2.2 Llama HF 格式（`convert_torch2transformers_llama`）

```python
llama_config = LlamaConfig(
    vocab_size=lm_config.vocab_size,
    hidden_size=lm_config.hidden_size,
    intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
    num_hidden_layers=lm_config.num_hidden_layers,
    num_attention_heads=lm_config.num_attention_heads,
    num_key_value_heads=lm_config.num_key_value_heads,
    ...
)
llama_model = LlamaForCausalLM(llama_config)
llama_model.load_state_dict(state_dict, strict=False)
llama_model.save_pretrained(transformers_path)
```

- 模型结构：标准 `LlamaForCausalLM`（HuggingFace 内置）
- 加载方式：`AutoModelForCausalLM.from_pretrained(path)` —— **不需要** `trust_remote_code=True`
- 兼容 **vLLM、llama.cpp、TGI、Text Generation Inference** 等所有支持 Llama 架构的工具

### 2.3 两种格式对比

| 维度 | MiniMind HF 格式 | Llama HF 格式 |
|------|-----------------|---------------|
| **模型类** | `MiniMindForCausalLM`（自定义） | `LlamaForCausalLM`（标准） |
| **加载** | 需 `trust_remote_code=True` | 无需特殊参数 |
| **兼容性** | 仅 MiniMind 项目生态 | 所有支持 Llama 的推理框架 |
| **config.json** | `"model_type": "minimind"` | `"model_type": "llama"` |
| **推荐用途** | 继续训练、调试 | 发布、部署、第三方工具集成 |

> **核心洞察**：MiniMind 是基于 LLaMA 架构的，所以它的 `state_dict` 的 key 命名与 `LlamaForCausalLM` 兼容。这就是为什么 `strict=False` 时能**部分匹配**加载成功。

---

## 第三章：三个转换函数详解

### 3.1 `convert_torch2transformers_minimind(torch_path, transformers_path, dtype)`

```
流程:
  1. 注册 AutoClass ──→ MiniMindConfig + MiniMindForCausalLM 注册到 HF Auto 机制
  2. 创建模型实例 ──→ MiniMindForCausalLM(lm_config) ← 用全局 lm_config
  3. 加载权重 ──→ torch.load(.pth) → load_state_dict(strict=False)
  4. 精度转换 ──→ .to(dtype)  (默认 float16)
  5. 保存模型 ──→ save_pretrained(transformers_path, safe_serialization=False)
  6. 保存 tokenizer ──→ AutoTokenizer.from_pretrained('../scripts/Model/') → save_pretrained()
  7. 修补 config ──→ 注入 tokenizer_class + extra_special_tokens
```

**`register_for_auto_class()` 的作用**：

```python
MiniMindConfig.register_for_auto_class()
MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
```

- 将 `MiniMindConfig` 注册到 `transformers` 的自动注册表中
- `save_pretrained` 时会在 `config.json` 中写入 `"auto_map": {"AutoConfig": "...", "AutoModelForCausalLM": "..."}` 字段
- 这样别人 `from_pretrained` 时，HF 会知道从哪个类加载

### 3.2 `convert_torch2transformers_llama(torch_path, transformers_path, dtype)`

```
流程:
  1. 构造 LlamaConfig ──→ 将 MiniMind 参数映射到 LlamaConfig 字段
  2. 创建 LlamaForCausalLM ──→ 标准 HF Llama 模型
  3. 加载权重 ──→ strict=False 忽略不匹配的 key
  4. 精度转换 + 保存模型 + tokenizer + 修补 config
```

**重点：`intermediate_size` 的计算公式**

```python
intermediate_size = 64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64)
```

这是 **SwiGLU** 架构决定的。回顾 `model_minimind.py`：

```
SwiGLU 的 FFN 有三个权重矩阵: gate_proj, up_proj, down_proj
标准 FFN 的 hidden 维度 = 4 * hidden_size
SwiGLU 的 hidden 维度 = 8/3 * hidden_size  ≈ 2.67 * hidden_size
  
公式分解:
  int(hidden_size * 8 / 3)        ← SwiGLU 的理论中间维度
  + 64 - 1                         ← 向上取整对齐到 64 的倍数
  // 64 * 64                       ← 64 对齐（某些 GPU 对非对齐维度有性能惩罚）
  
  以 hidden_size=512 为例:
  int(512 * 8/3) = 1365
  (1365 + 63) // 64 * 64 = 1428 // 64 * 64 = 22 * 64 = 1408
  所以 intermediate_size = 1408
```

### 3.3 `strict=False` 解决了什么问题？

```python
llama_model.load_state_dict(state_dict, strict=False)
```

MiniMind 的 `state_dict` 的 key（如 `model.layers.0.self_attn.q_proj.weight`）与 `LlamaForCausalLM` 的 key **不完全一致**：

| MiniMind state_dict key | LlamaForCausalLM key | 是否匹配 |
|------------------------|---------------------|---------|
| `model.layers.0.self_attn.q_proj.weight` | `model.layers.0.self_attn.q_proj.weight` | ✅ |
| `model.layers.0.self_attn.k_proj.weight` | `model.layers.0.self_attn.k_proj.weight` | ✅ |
| `model.layers.0.self_attn.v_proj.weight` | `model.layers.0.self_attn.v_proj.weight` | ✅ |
| `model.layers.0.self_attn.o_proj.weight` | `model.layers.0.self_attn.o_proj.weight` | ✅ |
| `model.layers.0.mlp.gate_proj.weight` | `model.layers.0.mlp.gate_proj.weight` | ✅ |
| `model.layers.0.mlp.up_proj.weight` | `model.layers.0.mlp.up_proj.weight` | ✅ |
| `model.layers.0.mlp.down_proj.weight` | `model.layers.0.mlp.down_proj.weight` | ✅ |
| `model.embed_tokens.weight` | `model.embed_tokens.weight` | ✅（weight tying） |
| `lm_head.weight` | `lm_head.weight` | ✅（与 embed_tokens 共享） |
| `model.norm.weight` | `model.norm.weight` | ✅ |
| `model.final_norm.weight` | ❌ | 没有对应的 key |

`strict=False` 允许**部分匹配**：匹配的 key 加载权重，不匹配的 key 保留初始化值，不抛异常。

> **注意**：如果 `LlamaForCausalLM` 中有 MiniMind 没有的 key（如 `lm_head.weight` 在 Llama 中独立存在，但 MiniMind 中与 `embed_tokens.weight` 共享），`strict=False` 也会忽略这些不匹配。这意味着加载后可能某些权重没有被正确赋值——验证方法是对比转换前后的模型输出。

### 3.4 `convert_transformers2torch(transformers_path, torch_path)`

```python
model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
```

- 反向转换：HF 格式 → `.pth`
- `model.state_dict()`：拿到所有参数（不包括缓存、优化器状态）
- `.cpu().half()`：转回 CPU + float16 精度
- 覆盖之前的 `.pth` 文件（如果路径相同）

---

## 第四章：关键细节分析

### 4.1 `safe_serialization=False`

```python
lm_model.save_pretrained(transformers_path, safe_serialization=False)
```

- `safe_serialization=True`（默认）：用 `safetensors` 格式保存（`.safetensors` 文件）
- `safe_serialization=False`：用 PyTorch `torch.save` 保存（`pytorch_model.bin` 文件）
- 这里显式设为 `False`，输出的是 `.bin` 文件

### 4.2 `tokenizer_config.json` 修补

```python
config_path = os.path.join(transformers_path, "tokenizer_config.json")
json.dump(
    {
        **json.load(open(config_path, 'r', encoding='utf-8')),
        "tokenizer_class": "PreTrainedTokenizerFast",
        "extra_special_tokens": {}
    },
    open(config_path, 'w', encoding='utf-8'),
    indent=2, ensure_ascii=False
)
```

- 从 `../scripts/Model/` 保存 tokenizer 后，读取其 `tokenizer_config.json`
- 注入 `tokenizer_class: "PreTrainedTokenizerFast"`——transformers 5.0 要求此字段
- 注入空的 `extra_special_tokens: {}`——也是 transformers 5.0 的兼容性要求
- 使用 `{**old_dict, **new_dict}` 语法合并（新值覆盖旧值）

这两行是**兼容 transformers 5.0+ 的必要步骤**，因为新版 transformers 对 tokenizer 配置的校验更严格。

### 4.3 精度处理

```python
lm_model = lm_model.to(dtype)  # dtype=torch.float16 (默认)
```

- 默认转换为 float16，权重文件缩小一半
- 可以改为 `torch.float32`（全精度）或 `torch.bfloat16`（如果硬件支持）
- 精度选择影响：
  - **推理速度**：float16 > bfloat16 > float32
  - **模型质量**：float32 ≈ bfloat16 > float16（极低精度下可能损失质量）
  - **文件大小**：float16 是 float32 的一半

### 4.4 `lm_config` 的位置问题

```python
if __name__ == '__main__':
    lm_config = MiniMindConfig(hidden_size=512, ...)
    ...
```

- `lm_config` 是一个**全局变量**（在 `if __name__` 块中定义）
- 三个转换函数都隐式引用了这个全局变量（`lm_config.hidden_size` 等）
- 这意味直接 import 这个模块时 `lm_config` 不存在，函数无法独立使用
- **设计缺陷**：应该把 `lm_config` 作为函数参数传入

---

## 第五章：`__main__` 入口分析

```python
if __name__ == '__main__':
    lm_config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, max_seq_len=8192, use_moe=False)
    torch_path = f"../models/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../MiniMind2-Small'
    convert_torch2transformers_llama(torch_path, transformers_path)
    # # convert transformers to torch model
    # convert_transformers2torch(transformers_path, torch_path)
```

**当前默认行为**：
- 转换 `full_sft_512.pth` → `../MiniMind2-Small/`（Llama 格式）
- MoE 和非 MoE 的文件名通过 `use_moe` 控制（`full_sft_512_moe.pth` 或 `full_sft_512.pth`）
- 反向转换被注释掉

**如果要转换 MoE 模型**：
```python
lm_config = MiniMindConfig(hidden_size=640, num_hidden_layers=8, max_seq_len=8192, use_moe=True)
torch_path = f"../models/full_sft_640_moe.pth"
transformers_path = '../MiniMind2-MoE'
```

**如果要转换其他阶段权重**：
```python
torch_path = "../models/dpo_512.pth"        # DPO 权重
# 或
torch_path = "../models/grpo_512.pth"       # GRPO 权重
```

---

## 第六章：与 `model_minimind.py` / `trainer_utils.py` 中 `init_model` 的对比

| 维度 | convert_model.py | trainer_utils.py / serve_openai_api.py |
|------|-----------------|--------------------------------------|
| **加载方式** | `MiniMindForCausalLM(config)` + `load_state_dict` | 同左（原生 .pth 分支） |
| **strict 参数** | `strict=False` | `strict=True`（训练脚本要求严格匹配） |
| **目的** | 格式转换 | 训练/推理 |
| **精度** | 转为 float16 | 保持训练精度 |
| **特殊处理** | 注册 AutoClass、修补 tokenizer_config | 加载 LoRA、初始化分布式 |

---

## 第七章：动手练习

### 基础练习

1. **运行转换**：执行 `python scripts/Tools/convert_model.py`，查看输出的 `../MiniMind2-Small/` 目录结构

2. **验证加载**：编写 Python 脚本，用 `AutoModelForCausalLM.from_pretrained` 加载转换后的模型，输入一段文本测试推理是否正常

3. **对比大小**：比较 `.pth` 文件（float32）和转换后的 `pytorch_model.bin`（float16）的大小差异

### 进阶练习

4. **修改 `__main__` 支持命令行参数**：用 `argparse` 让用户指定 `--hidden_size`、`--weight`（pretrain/full_sft/dpo 等）、`--use_moe`，而不是硬编码

5. **修改 `__main__` 支持 `--format` 选择**：让用户用 `--format minimind` 或 `--format llama` 选择输出格式

6. **添加 `--dtype` 参数**：支持 float32 / float16 / bfloat16 三种精度选择

7. **将 `lm_config` 改为函数参数**：解耦全局变量，让三个转换函数可以独立调用

### 深入练习

8. **输出一致性验证**：对同一组输入，对比转换前后模型的输出 logits 是否一致（用 `allclose`），验证 `strict=False` 没有丢失关键权重

9. **添加 MoE 兼容**：当前 `convert_torch2transformers_llama` 不支持 MoE（LlamaForCausalLM 没有 MoE 结构）。为 `convert_torch2transformers_minimind` 添加 MoE 支持，并确保 `register_for_auto_class` 后可以正常加载

10. **safetensors 支持**：将 `safe_serialization` 改为可配置，支持输出 `.safetensors` 格式（更安全、更快加载）

11. **批量转换脚本**：写一个脚本遍历 `models/` 目录下的所有 `.pth` 文件，批量转换为 Llama HF 格式，每个权重保存到独立的目录

---

## 自测题

1. **`register_for_auto_class()` 是做什么的？不调用会怎样？**

2. **为什么 `load_state_dict` 要用 `strict=False`？`strict=True` 时可能抛出什么错误？**

3. **`intermediate_size` 的计算公式 `64 * ((int(hidden_size * 8 / 3) + 64 - 1) // 64)` 中，`+ 64 - 1` 和 `// 64` 的作用是什么？**

4. **MiniMind 格式和 Llama 格式的转换结果有什么区别？各有什么适用场景？**

5. **为什么转换后要手动修补 `tokenizer_config.json`？不修补会有什么问题？**

6. **`convert_torch2transformers_minimind` 比 `convert_torch2transformers_llama` 多了什么步骤？为什么？**

7. **`model.state_dict()` 和 `torch.save(model, path)` 有什么区别？**

8. **`lm_config` 当前是全局变量，这个设计有什么缺陷？怎么改进？**

9. **转换后 `pytorch_model.bin` 比原始 `.pth` 文件小多少？为什么？**

10. **`safe_serialization=False` 和 `safe_serialization=True`（safetensors）有什么区别？**

---

## 自测题参考答案

<details>
<summary>点击展开参考答案</summary>

### Q1: `register_for_auto_class()` 的作用

调用后，`save_pretrained` 会在 `config.json` 中写入 `auto_map` 字段，记录自定义类和 `AutoModel` 的映射关系。这样别人用 `AutoModelForCausalLM.from_pretrained(path)` 时，HF 自动注册机制会知道要去加载 `MiniMindForCausalLM`。如果不调用，`config.json` 中没有 `auto_map`，`from_pretrained` 会不知道用哪个类，导致加载失败。

### Q2: `strict=False` 的原因

因为 `state_dict` 中的 key 与模型参数并不完全一一对应（如 weight tying 导致 `lm_head.weight` 被共享，有些 key 名不同或缺失）。`strict=True` 要求**完全匹配**——每个 key 都能找到对应参数且不能有多余的 key，会抛出 `RuntimeError: Missing key(s) ...` 或 `Unexpected key(s) ...`。`strict=False` 允许部分匹配，遇到不匹配的 key 时只打印警告。

### Q3: 64 对齐的意义

- `+ 64 - 1`：向上取整的"加数"，确保整数除法时不会向下取整
- `// 64`：除以 64 取整
- `* 64`：乘以 64 恢复
- 总效果：**向上取整到 64 的倍数**
- 原因是某些 GPU 计算（特别是 Tensor Core）对 64 或 128 对齐的维度有性能优势

### Q4: 两种格式

见第二章 2.3 的对比表。

### Q5: tokenizer_config 修补

Transformers 5.0+ 要求 `tokenizer_config.json` 必须包含 `tokenizer_class` 和 `extra_special_tokens` 字段。旧版本的 tokenizer 保存时可能没有这两个字段。不加的话，在新版 transformers 中加载 tokenizer 时会报兼容性错误。

### Q6: `convert_torch2transformers_minimind` 的额外步骤

多了 `register_for_auto_class()` 调用，因为 `MiniMindForCausalLM` 是自定义类，HF 的 Auto 机制不知道它的存在，需要主动注册。而 `LlamaForCausalLM` 是 HF 内置的标准类，不需要注册。

### Q7: `state_dict()` vs `torch.save(model, path)`

`model.state_dict()` 只返回模型参数（`OrderedDict` 包含权重张量），不包含模型结构信息。`torch.save(model, path)` 保存整个模型对象（包括结构、配置等），但需要模型类定义可序列化。通常推荐只保存 `state_dict`（更轻量、更安全）。

### Q8: 全局变量 `lm_config` 的缺陷

- 函数隐式依赖全局变量，不能独立调用
- 不同函数可能需要不同的 config（如 MoE vs 非 MoE）
- 无法并行/批量转换（全局变量会被覆盖）
- 改进方案：将 `lm_config` 作为函数参数传入

### Q9: 文件大小差异

原始 `.pth` 通常以 float32（4 字节/参数）保存，转换后 `pytorch_model.bin` 以 float16（2 字节/参数）保存。所以理论上缩小约 **50%**。实际因为 weight tying（`lm_head.weight` 和 `embed_tokens.weight` 共享同一张量，只存一次），大小会略有不同。

### Q10: `.bin` vs `.safetensors`

- `.bin`：基于 PyTorch `torch.save`，格式灵活但存在 pickle 安全风险（可执行任意代码）
- `.safetensors`：HuggingFace 推出的安全格式，只存储张量数据，无 pickle 风险，加载速度更快（支持零拷贝、多线程读取）
- 两者在存储的数值上完全等价

</details>

---

## 拓展阅读

- [HuggingFace 文档：分享自定义模型](https://huggingface.co/docs/transformers/main/en/custom_models)
- [HuggingFace 文档：safetensors](https://huggingface.co/docs/safetensors/index)
- [model_minimind.py 学习计划](./model_minimind_study_plan.md) — 模型架构详解
- [LlamaForCausalLM 源码](https://huggingface.co/docs/transformers/model_doc/llama)
