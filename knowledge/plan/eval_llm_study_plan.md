# eval_llm.py 学习计划指引

## 一、文件定位

`eval_llm.py` 是 MiniMind 项目的**推理与对话入口**，负责加载已训练好的权重并与之交互。它不参与训练，只做模型加载 → 输入处理 → 生成推理 → 输出展示这条链路。

## 二、前置知识

| 概念 | 建议学习途径 |
|------|-------------|
| Python `argparse` 命令行参数解析 | Python 官方文档 |
| PyTorch `model.eval()` / `no_grad` 推理模式 | PyTorch 官方教程 |
| HuggingFace `transformers` 库（`AutoTokenizer`, `AutoModelForCausalLM`, `TextStreamer`） | HuggingFace 文档 |
| Chat Template（`apply_chat_template`） | HuggingFace 博文 |
| LLM 生成参数（`temperature`, `top_p`, `repetition_penalty`） | 多篇博客（如 HF 生成策略指南） |
| Attention Mask | Transformer 论文 / HF 文档 |

## 三、文件逐段精读

### 第 1 层：import & 全局（L1–L12）

```python
from scripts.Model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.Model.model_lora import *
from scripts.Trainer.trainer_utils import setup_seed, get_model_params
```

**要点**：
- 该项目有两个模型入口：原生的 `MiniMindForCausalLM`（自定义 torch 权重）和 HF 的 `AutoModelForCausalLM`（transformers 格式）。
- `model_lora.py` 导出 `apply_lora` / `load_lora`，LoRA 注入发生在推理前。

**思考题**：为什么需要 `warnings.filterwarnings('ignore')`？去掉会怎样？
    答：屏蔽 transformers/PyTorch 在运行时的各类警告，不让它们打印到终端。
      警告多的原因不是 eval_llm.py 代码写得差——而是 HuggingFace transformers
      库本身版本迭代快、接口弃用频繁，加载模型和 tokenizer 时会自动弹出
      deprecation warning、配置未显式设置的提醒等。几乎所有用 transformers
      的项目都有这个问题，常见做法就是直接 ignore。源码 L12 注释：
      "忽略代码运行过程中的警告信息（让终端输出更干净）"。
      去掉后终端的模型回复会被大量 UserWarning/FutureWarning 淹没，影响观察输出。

---

### 第 2 层：`init_model()`（L14–L53）

**核心逻辑分支**：

```
load_from
 ├─ 包含 "model" → 加载自定义 MiniMind 权重（.pth 文件）
 │   ├─ 根据 hidden_size / num_hidden_layers / use_moe 实例化配置
 │   ├─ 从 ./out/{weight}_{hidden_size}[_moe].pth 读取 state_dict
 │   └─ 若指定 lora_weight → 注入 LoRA 并加载 LoRA 权重
 └─ 不包含 "model" → 用 transformers 加载 HF 格式权重
```

**关键知识点**：
- `state_dict` 是什么？`load_state_dict(strict=True)` 的 strict 参数含义？
- `model.eval()` 做了什么（关闭 dropout / batchnorm 等）？
- `model.to(device)` 是 in-place 还是返回新对象？
- 权重命名规则（`{weight}_{hidden_size}{_moe}.pth`）和 `--weight` 各选项的对应关系（pretrain / full_sft / rlhf / reason / ppo_actor / grpo / spo）。

**建议并行阅读**：`scripts/Model/model_minimind.py`（模型结构）、`scripts/Model/model_lora.py`（LoRA 注入细节）。

---

### 第 3 层：`main()` 入口（L55–L96）

**argparse 参数全表**：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--load_from` | `model` | 选择加载路径（原生权重 vs HF 格式） |
| `--save_dir` | `out` | 权重存放目录 |
| `--weight` | `full_sft` | 权重阶段选择 |
| `--lora_weight` | `None` | 可选 LoRA 权重名 |
| `--hidden_size` | `512` | 模型维度（定模型大小） |
| `--num_hidden_layers` | `8` | Transformer 层数 |
| `--use_moe` | `0` | 是否启用 MoE |
| `--inference_rope_scaling` | `False` | 是否启用 RoPE 外推 |
| `--max_new_tokens` | `8192` | 最大生成长度 |
| `--temperature` | `0.85` | 采样温度 |
| `--top_p` | `0.85` | Nucleus 采样 |
| `--historys` | `0` | 多轮对话历史轮数 |
| `--show_speed` | `1` | 显示生成速度 |
| `--device` | `cuda/cpu` | 运行设备 |

**思考题**：`--historys=0` 和 `--historys=2` 在执行逻辑上有什么区别？（参考 L137）

---

### 第 4 层：对话循环（L98–L186）

#### 4.1 预设 prompts（L99–L108）

8 个覆盖**常识问答、代码生成、科学解释、推荐**等场景的测试 prompt。

#### 4.2 模式选择（L117–L124）

```python
input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
```

- 模式 0：遍历预设列表，适合批量回归测试。
- 模式 1：`iter(lambda: input(...), '')` 不断读取终端输入，输入空行则终止。

#### 4.3 Chat Template 与 Tokenization（L142–L154）

```python
templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
if args.weight == 'reason':
    templates["enable_thinking"] = True

inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
```

**关键知识点**：
- **pretrain vs SFT 模型**的输入差异：pretrain 没有 chat template，直接拼 BOS 喂原始文本。
- **reason 模型**额外开启了 `enable_thinking`（对应 DeepSeek 式思维链格式）。
- **对话历史截取**（L137）：`conversation[-args.historys:]`，偶数轮才能保持 user/assistant 完整配对。

#### 4.4 模型生成（L160–L171）

```python
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    streamer=streamer,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    top_p=args.top_p,
    temperature=args.temperature,
    repetition_penalty=1.0,
)
```

**核心参数作用**：
- `do_sample=True` → 启用随机采样（否则 greedy decoding）。
- `temperature` → 控制概率分布的尖锐程度（越低越确定性）。
- `top_p` → 只保留累积概率 p 以内的 token。
- `repetition_penalty` → 对已出现 token 的 logit 进行惩罚。
- `streamer` → 边生成边打印，实现打字机效果。

**思考题**：`pad_token_id` 和 `eos_token_id` 不设置会怎样？`repetition_penalty=1.0` 意味着什么？
   答：
      - `eos_token_id` 不设置：模型不知道结束符是什么，不会主动停，一直生成直到
        max_new_tokens 上限，输出尾部全是无意义内容。
      - `pad_token_id` 不设置：generate() 内部默认取 eos_token_id 当 pad_token_id，
        但 eos 和 pad 在语义上不同，可能导致生成提前误终止（模型把填充位当成序列结束）。
        它跟 attention_mask 不重叠——attention_mask 控制注意力计算时忽略哪些位置（计算层），
        pad_token_id 告诉生成循环哪个 token ID 算填充（生成控制层），两者在不同阶段起作用。
        在 eval_llm.py 的单条推理场景下影响小，但不设会弹 warning。
       - `repetition_penalty=1.0`：不做任何重复惩罚，已出现 token 的 logit 完全不受影响。
    【追问：生成循环为什么要用 pad_token_id 来填充？难道模型会预测出 pad_token 这个 ID 吗？】
      不是模型预测出 pad_token。pad_token_id 在 generate() 中用于 batch 生成：
      当 batch 内多个序列同时生成时，先输出 EOS 结束的序列后面需要填充一些占位 token
      来对齐长度，其他序列才能继续并行算。pad_token_id 就是指定用哪个 token ID 来填充，
      同时 attention_mask 对应位置置 0，注意力计算直接跳过这些填充位。
      pad_token_id 不设的后果：generate() 拿 eos_token_id 当默认 pad。问题在于——
      如果模型恰好在某一步预测出了等于 eos_token_id 的 token，生成循环分不清
      "这是序列真的结束了"还是"这只是个填充位"，可能导致提前误终止。
      在 eval_llm.py 单条推理场景下没有 batch，用不到它，传它只是为 API 完整不弹 warning。
#### 4.5 响应处理与速度统计（L174–L186）

```python
response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
```

**关键点**：通过 `generated_ids` 与 `input_ids` 的长度差计算新生成的 token 数量和对应的 tokens/s 速度。

---

## 四、项目关联文件图谱

```
eval_llm.py
 ├─ scripts/Model/model_minimind.py      ← 模型定义（MiniMindForCausalLM）
 ├─ scripts/Model/model_lora.py           ← LoRA 注入逻辑
 ├─ scripts/Trainer/trainer_utils.py      ← setup_seed, get_model_params
 ├─ scripts/Deploy/serve_openai_api.py   ← OpenAI 兼容 API 版推理（相近逻辑）
 ├─ scripts/Trainer/train_full_sft.py     ← full_sft 训练脚本
 ├─ scripts/Trainer/train_pretrain.py     ← pretrain 训练脚本
 ├─ scripts/Trainer/train_ppo.py          ← RLHF-PPO 训练脚本
 ├─ scripts/Trainer/train_grpo.py         ← GRPO 训练脚本
 ├─ scripts/Trainer/train_spo.py          ← SPO 训练脚本
 └─ scripts/Trainer/train_reason.py       ← Reason 训练脚本
```

**建议学习顺序**：先掌握 `eval_llm.py`（推理入口）→ 再按 `pretrain → full_sft → reason → ppo/grpo/spo` 顺序学习训练脚本，每个阶段都有对应 `--weight` 选项。

---

## 五、动手练习

### 基础

**1. 运行 `python scripts/Deploy/eval_llm.py --weight pretrain` 观察输出与 `full_sft` 的差异**

```bash
# 激活环境，自动测试模式（echo "0" 表示选择 0-自动测试）
conda activate minimind
echo "0" | python scripts/Deploy/eval_llm.py --weight pretrain
```

pretrain 输出示例（prompt="你有什么特长？"）：
```
我是一台机器人，没有具体的特长。但是，如果您想了解有关特定主题的信息...
```
随后开始大量重复无意义内容（"祝！谢谢。如果您...祝！祝！"循环），不具备对话能力。

对比 `--weight full_sft` 同一 prompt 的输出：
```
我是一台机器学习模型，通过深度学习技术不断学习和优化我的回答。
我能够从大量数据中学习和提取有用的信息，帮助我进行预测和决策。
```

**结论**：pretrain 只在纯文本上做过自回归语言建模，没学过对话格式；full_sft 经过指令微调后学会了结构化的回答。

---

**2. 将 `temperature` 分别设为 0.1 / 0.85 / 1.5 对比生成质量**

```bash
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --temperature 0.1
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --temperature 0.85
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --temperature 1.5
```

**temperature=0.1**（prompt="你有什么特长？"）：
```
我被设计用来回答各种问题、提供信息、执行任务或执行任务。
我被设计用来帮助用户解决问题、提供信息、进行对话、
进行娱乐、进行教育、进行科学研究、进行数据分析、
进行数据分析、进行数据分析、进行数据分析...
```
→ 极其重复，卡在"进行数据分析"死循环

**temperature=0.85（默认）**（prompt="请用Python写一个计算斐波那契数列的函数"）：
```python
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```
→ 代码基本正确，有结构

**temperature=1.5**（同一斐波那契 prompt）：
```
斐波那契数列是一个数学序列，其中每当一个数字超过它的某个值时，
它会增加1和1开始计算次数之比。所以，数列中的每一个数都增加了1和1。
```
→ 胡言乱语，未能给出正确代码

---

**3. 将 `top_p` 设为 0.5 / 0.9 / 1.0 观察效果变化**

```bash
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --top_p 0.5
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --top_p 0.9
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --top_p 1.0
```

**top_p=0.5**（候选集小）：
- "你有什么特长" → 简短且集中
- "光合作用" → 严重重复，"光能转化"要点重复了 8 遍
- "比较猫和狗" → 大量重复（"温顺"反复出现），内容空洞
- 特点：容易陷入局部重复，长文本质量差

**top_p=0.9**（候选集大，接近默认值）：
- "解释什么是机器学习" → 最全面的回答，分 8 步详细说明
- "比较猫和狗" → 结构较完整，优缺点都有展开
- 特点：大部分回答质量好，信息量充足

**top_p=1.0**（不截断，保留全部候选 token）：
- "斐波那契数列" → 代码正确但测试代码有重复行
- "比较猫和狗" → 结构完整但部分内容偏离
- "推荐美食" → 与 0.9 几乎一样（简单任务差别不大）
- 特点：与 0.9 差异不明显，复杂任务边际收益很小

**对比总结**：

| top_p | 候选集 | 重复倾向 | 信息量 | 适用场景 |
|-------|--------|---------|--------|---------|
| 0.5 | 小 | 高（长文本易死循环） | 少 | 简单问答，需要高确定性 |
| 0.85（默认） | 适中 | 低 | 高 | 通用场景的最佳平衡 |
| 0.9 | 较大 | 低 | 高 | 需要更丰富细节时 |
| 1.0 | 全部 | 低 | 中 | 边际收益很小，极少用 |

### 进阶

**4. 修改 `prompts` 列表，添加与当前 `--weight` 对应的领域测试 prompt**

在 `eval_llm.py` 中扩展 prompts，根据 `--weight` 追加领域相关测试 prompt：

```python
prompts = [
    '你有什么特长？',
    '为什么天空是蓝色的',
    # ... 默认 8 个通用 prompt ...
]

if args.weight == 'reason':
    prompts += [
        '小明有5个苹果，给了小红2个，又买了3个，现在小明有几个苹果？',
        '一个三角形两边长分别为3和4，求第三边的长度（已知是直角三角形）',
    ]
elif 'medical' in args.lora_weight:
    prompts += [
        '感冒和流感有什么区别？',
        '如何预防高血压？',
    ]
```

**作用**：reason 模型自动追加数学推理题，medical LoRA 自动追加医学问答，无需手动拼接参数。

---

**5. 修改 LoRA 加载逻辑，让 `lora_weight` 支持多个 LoRA 合并**

修改 `scripts/Model/model_lora.py`，新增 `apply_lora_multi()` 和 `load_lora_multi()`：

- `apply_lora_multi(model, ranks)`：为每层注入多个 LoRA 模块（每个模块一个 rank），前向传播时把多个 LoRA 分支输出求和再与原路相加
- `load_lora_multi(model, paths)`：从多个 .pth 文件加载权重到对应 LoRA 模块，支持 `merge_weights` 缩放系数

`eval_llm.py` 中 `--lora_weight` 支持逗号分隔：

```bash
# 单个 LoRA（与原行为一致）
python scripts/Deploy/eval_llm.py --weight full_sft --lora_weight lora_medical

# 多个 LoRA 合并推理
python scripts/Deploy/eval_llm.py --weight full_sft --lora_weight lora_identity,lora_medical
```

**多 LoRA 合并原理**：
```
output = Wx + Σ(α_i · B_i(A_i(x)))
```
每个 LoRA 分支独立计算低秩增量，求和后叠加到原始输出上。多个 LoRA 的效果可以叠加（身份 + 领域知识），权重通过 `merge_weights` 调节各分支占比。

**【追问：多个独立训练的 LoRA 合并到一起，不会让输出乱掉吗？】**

这是个好问题。多 LoRA 合并能工作的前提是：**LoRA 的增量是加在权重空间上的（W + ∆W₁ + ∆W₂），这是线性叠加**。能不能正常工作取决于两个条件：

**① LoRA 捕获的能力是正交/独立的**
如果 LoRA₁ 学的是"角色身份"（说话风格），LoRA₂ 学的是"医学知识"（专业术语），它们更新的是权重矩阵中不同的子空间方向。加起来相当于"用医学知识的语气说话"——两种能力可以共存，不会冲突。

**② 共用同一个基座模型**
所有 LoRA 必须基于同一个 `full_sft_512.pth` 训练出来的。如果基座模型版本不同（权重 W 不同），∆W₁ 和 ∆W₂ 的"参考零点"不一样，加起来没有意义。

**什么情况下会乱？**
- **行为冲突**：LoRA₁ 学"说话简洁"，LoRA₂ 学"说话详细"——两者在权重空间的方向相反，加起来相互抵消，输出变得不伦不类
- **数据冲突**：LoRA₁ 在医学数据上训练，LoRA₂ 在娱乐数据上训练，但两者对同一个 token 的 logit 调整方向相反——最终输出可能偏向某一方或两者都不像
- **线性假设不成立**：LoRA 的增量是低秩的（`∆W = BA`），两个低秩矩阵相加仍然是低秩的，但如果两者都修改了同一组奇异方向，叠加后效果不是简单的"能力相加"

**实际使用建议**：
- 推荐合并互补型 LoRA（如 identity + domain），不推荐合并冲突型 LoRA（如 style_A + style_B）
- 用 `merge_weights` 调节各分支权重（如 identity:0.3 + medical:0.7），通过实验找到最佳比例
- 如果合并后效果不好，可以对齐到 calibration 数据上搜索最优 merge 权重

---

**6. 增加 `--repetition_penalty` 参数，测试其对生成的影响**

在 `eval_llm.py` 中添加参数（之前写死为 1.0）：

```python
parser.add_argument('--repetition_penalty', default=1.0, type=float,
    help="重复惩罚系数，>=1.0。1.0=不惩罚，1.1=轻微压制，2.0=强惩罚")
```

生成调用处改为：`repetition_penalty=args.repetition_penalty`

```bash
# 测试不同惩罚强度
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --repetition_penalty 1.0
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --repetition_penalty 1.2
echo "0" | python scripts/Deploy/eval_llm.py --weight full_sft --repetition_penalty 2.0
```

**repetition_penalty=1.2 测试结果**（prompt="比较一下猫和狗作为宠物的优缺点"）：
```
猫是肉食性动物，通常被认为是杂食性动物。它们在生态系统中扮演着多种角色：
1. 提供食物：猫可以分成两类...
2. 提供娱乐价值：有些猫甚至可以在室内生活得很好...
```
对比默认 `1.0` 时同一 prompt 会大量重复"温顺"，`1.2` 后重复明显减少，内容多样性增加，但"猫和狗"原本对等的对比结构被打破（只讲了猫，忽略了狗）。

**repetition_penalty=2.0 测试结果**（同一 prompt）：
```
好的，我来帮你分析。首先是关于它们的健康问题：不同品种的猫有不同的体型、毛发类型
以及性格特点等因素对其行为的影响程度也不尽相同...
```
惩罚过强导致模型开始混入英文、数学表达式等不相关内容。强力抑制已出现 token 后，
模型被迫选择低频 token，输出变得混乱，生成质量严重下降。

**惩罚机制**（HuggingFace 实现）：
```
if score < 0:  score *= repetition_penalty（负 logit 绝对值变大，更负）
if score ≥ 0:  score /= repetition_penalty（正 logit 被缩小）
```
已出现 token 的 softmax 概率下降 → 模型更倾向于选新词。

**对比总结**：

| repetition_penalty | 重复程度 | 输出质量 | 适用场景 |
|-------------------|---------|---------|---------|
| 1.0（默认） | 可能较多重复 | 正常 | 通用场景，不做额外惩罚 |
| 1.2 | 重复减少 | 较好 | 需要抑制重复时 |
| 2.0 | 几乎不重复 | 差（胡言乱语） | 极端情况，不推荐 |

### 深入

**7. 阅读 `model_minimind.py` 中的 `generate()` 方法，理解其中如何调用 `streamer`**

`MiniMindForCausalLM`（`model_minimind.py:2540`）继承自 `GenerationMixin`，**没有自定义 `generate()` 方法**。`generate()` 直接来自 HuggingFace 的 `GenerationMixin`。

**streamer 的调用链路**（从 eval_llm.py 到 transformers 内部）：

```
eval_llm.py L201:
  streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

eval_llm.py L469-496:
  model.generate(..., streamer=streamer)
      ↓
  GenerationMixin.generate()  ← transformers 内置方法
      ↓ 每生成一个 token 调用一次
  streamer.put(next_token_id)
      ↓
  TextStreamer 内部解码 token_id → 文本 → 打印到终端
      ↓ 生成结束
  streamer.end()
```

**streamer 的核心作用**：边生成边打印，实现"打字机"效果。如果不传 streamer，generate() 会等全部生成完后一次性返回，用户需要等待整个回复生成完才能看到内容。

**CustomStreamer（serve_openai_api.py）为什么要这样设计？**

```python
class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue

    def on_finalized_text(self, text, stream_end=False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)
```

原因：**`model.generate()` 是同步阻塞调用**，在 API 服务器中直接调用会阻塞整个请求线程，无法实现流式推送。设计思路：

1. `model.generate(streamer=streamer)` 在**后台子线程**中运行（`Thread(target=_generate).start()`）
2. `TextStreamer` 默认的 `on_finalized_text` 是把解码后的文本直接 `print` 到终端——这在 API 场景下没用，需要改为通过 queue 传递给主线程
3. `CustomStreamer` 重写 `on_finalized_text`，每生成一段文本就 `queue.put(text)`，生成结束时 `queue.put(None)` 作为终止信号
4. 主线程（FastAPI 的异步生成器）从 queue 中不断读取文本，通过 SSE（Server-Sent Events）逐块推送回客户端

**对比 eval_llm.py**：交互式终端不需要 queue——直接用 `TextStreamer` 的默认行为 `print` 到终端就行。API 服务器的输出目的地不是终端，而是 HTTP 响应流，所以需要 queue + 后台线程来桥接同步的 `generate()` 和异步的 SSE 推送。

---

**8. 对比 `scripts/Deploy/serve_openai_api.py` 中的 `init_model` 与 `eval_llm.py` 的 `init_model`**

两边的 `init_model` 核心逻辑几乎相同（都是 `'model' in load_from` 分支判断），差异点：

| 差异项 | eval_llm.py | serve_openai_api.py |
|--------|-------------|-------------------|
| ckp 路径 | `./{save_dir}/...` | `../{save_dir}/...`（scripts/ 子目录运行） |
| config 参数 | 无 `max_seq_len` | 有 `max_seq_len` 参数 |
| 参数量打印 | `get_model_params()` | 直接 `print(sum(...))` |
| torch.load | `weights_only=False`（已修复） | 无 `weights_only`（旧代码） |
| 多 LoRA 合并 | ✅ 支持逗号分隔 | ❌ 不支持 |
| 参数列表 | 完整（temperature/top_p/repetition_penalty 等） | 精简（只有核心参数） |

**思考原因**：serve_openai_api.py 是 API 服务端，设计上追求轻量和稳定，不包含学习/实验性参数（如 `show_speed`、`historys`），也不需要在推理时反复修改 LoRA 合并策略。

---

**9. 如果要支持 `--quantize`（量化推理），需要在 `init_model` 的哪个位置插入量化逻辑？**

同 eval_llm.py 自测 Q16 的分析：

- **权重量化（省显存）**：`model.load_state_dict()` 之后插入，把每层 Linear 的 float 权重替换为 4bit 表示（如 bitsandbytes 的 `Linear4bit`）。不需要额外数据，加载时就完成转换。
- **激活值量化（加速推理）**：无法在 `init_model` 中一步完成。激活的 scale/zero_point 依赖实际输入分布，需要跑 calibration 数据集收集统计量后才能确定量化参数（PTQ 流程）。

插入位置伪代码：
```python
def init_model(args):
    # ... 现有加载逻辑 ...
    model.load_state_dict(torch.load(...), strict=True)
    
    if args.quantize:
        # 权重量化：替换 Linear 为量化版本
        from bitsandbytes.nn import Linear4bit
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quant_layer = Linear4bit(...)  # 4bit 量化
                # 替换原层
        # 注意：激活量化需要额外的 calibration 步骤
    
    return model.eval().to(args.device), tokenizer
```

---

## 六、学习目标检查清单

- [✓] 能画出 `init_model` 的两个分支流程图
- [✓] 能说出 `temperature`、`top_p`、`repetition_penalty` 各自的控制维度
- [✓] 能解释 pretrain 权重输入时为什么跳过 `apply_chat_template`
- [✓] 能说明 `historys` 参数的截取逻辑及为什么必须是偶数
- [✓] 能区分 `model.eval()` 和 `torch.no_grad()` 的异同
- [✓] 能复述权重文件的命名规则 `{weight}_{hidden_size}[_moe].pth`
- [✓] 能在 `eval_llm.py` 基础上给模型增加新的命令行参数
