# web_demo.py 学习计划

> **文件位置**: `scripts/Deploy/web_demo.py`（328 行）
> **角色**: 基于 Streamlit 的交互式 Web 聊天界面，支持本地模型和 API 两种模式
> **前置知识**: 已学完 `eval_llm.py`（对话流程）、`serve_openai_api.py`（API 服务端）、`chat_openai_api.py`（API 客户端）

---

## 文件全景图

```
web_demo.py
│
├── CSS 样式注入（st.markdown + 圆形按钮等）
├── process_assistant_content()    ← 渲染 <think> 推理内容
├── load_model_tokenizer()         ← @st.cache_resource 缓存模型
├── clear_chat_messages()          ← 清除对话
├── init_chat_messages()           ← 初始化/恢复历史
├── regenerate_answer()            ← 重新生成
├── delete_conversation()          ← 删除某轮对话
│
├── 侧边栏设置（st.sidebar）
│   ├── history_chat_num 滑块      ← 历史轮数
│   ├── max_new_tokens 滑块         ← 最大生成长度
│   ├── temperature 滑块            ← 采样温度
│   └── model_source 单选框         ← 本地模型 / API
│
├── main()
│   ├── 模式判断 → 加载模型或设 None
│   ├── 渲染历史消息 + 删除按钮
│   ├── 接收用户输入
│   ├── API 模式 → OpenAI SDK 流式调用
│   └── 本地模式 → TextIteratorStreamer + Thread
│
└── __main__
    └── 导入 transformers → 调用 main()
```

---

## 第一章：运行方式

### 1.1 环境准备

```bash
# 安装 streamlit（如尚未安装）
pip install streamlit

# 验证安装
streamlit --version
```

### 1.2 两种运行模式

#### 模式 A：本地模型模式（推荐体验）

直接加载 HuggingFace 格式的模型权重进行推理，无需启动额外服务。

```bash
# 从项目根目录运行
cd /mnt/data_2t_0/Projects/minimind
streamlit run scripts/Deploy/web_demo.py

# 或指定端口
streamlit run scripts/Deploy/web_demo.py --server.port 8501
```

启动后在浏览器打开 `http://localhost:8501`。

侧边栏选择"本地模型"→ 从下拉菜单选一个模型即可对话。

**前置条件**：需要有 HuggingFace 格式的模型权重（下载到项目外或通过 `convert_model.py` 转换）。`MODEL_PATHS` 中默认的路径如 `../MiniMind2` 是相对于 `scripts/` 目录的上级目录。

#### 模式 B：API 模式

先启动 API 服务，再启动 Web 界面，两者通过 HTTP 通信。

```bash
# 终端 1：启动 API 服务（在项目根目录运行）
python scripts/Deploy/serve_openai_api.py --weight full_sft --hidden_size 512

# 终端 2：启动 Web 界面（在项目根目录运行）
streamlit run scripts/Deploy/web_demo.py
```

Web 界面侧边栏选择"API"→ 填入 API URL（默认 `http://127.0.0.1:8000/v1`）、Model ID（默认 `minimind`）、Model Name（默认 `MiniMind2`）→ 开始聊天。

> **注意**：`serve_openai_api.py` 默认端口是 8998，而 web_demo.py 的 API URL 默认值是 `http://127.0.0.1:8000/v1`。使用时需要两者匹配。推荐在 web_demo.py 侧边栏改填 `http://127.0.0.1:8998/v1`，或给 serve_openai_api.py 传 `--port 8000`（当前版本不支持 `--port` 参数，直接改代码中的 `port=8998` 为 `port=8000`）。

### 1.3 界面元素速览

| 区域 | 元素 | 说明 |
|------|------|------|
| 顶部 | Logo + 标语 | ModelScope 在线图片 |
| 侧边栏 | History 滑块 | 0~6 步长 2，控制历史轮数 |
| 侧边栏 | Max Length 滑块 | 256~8192，控制最大生成长度 |
| 侧边栏 | Temperature 滑块 | 0.6~1.2，步长 0.01 |
| 侧边栏 | 模型来源单选框 | 本地模型 / API |
| 侧边栏 | 模型选择下拉框 | 仅本地模式下出现 |
| 侧边栏 | API 配置输入框 | 仅 API 模式下出现 |
| 主区域 | 对话气泡 | 用户右对齐灰色，助手左对齐带头像 |
| 主区域 | 删除按钮 × | 每个助手消息旁的圆形按钮 |
| 底部 | 输入框 | chat_input 组件 |

---

## 第二章：初始化与样式

```python
st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")
```

### 2.1 `set_page_config`

- Streamlit 必须第一个调用的配置函数
- `page_title`：浏览器标签页标题
- `initial_sidebar_state="collapsed"`：默认收起侧边栏，节省主区域空间

### 2.2 CSS 样式注入

使用 `st.markdown("""<style>...""", unsafe_allow_html=True)`：

- `.stButton button`：将按钮变成圆形（`border-radius: 50%`）
- 第一段样式设置 32px 大小的按钮（删除用 🗑）
- 第二段样式用 `all: unset` 重置基础按钮，再设为 18px 小圆点（× 删除按钮）
- `.stMainBlockContainer > div:first-child`：调整顶部间距
- `.stApp > div:last-child`：调整底部间距

> **为什么两组按钮样式？** 第一段 `.stButton button` 是针对 `st.button("🗑")` 的 32px 按钮，第二段 `all: unset` 更激进地重置，是针对 `st.button("×")` 的 18px 迷你按钮。但是 **CSS 层叠导致后面覆盖前面**——`all: unset` 会重置所有属性，实际上 32px 的样式永远不会生效。这是一个设计上的小 bug。

---

## 第三章：推理内容渲染

```python
def process_assistant_content(content):
```

### 3.1 条件判断

```python
if model_source == "API" and 'R1' not in api_model_name:
    return content
if model_source != "API" and 'R1' not in MODEL_PATHS[selected_model][1]:
    return content
```

- 仅当模型名包含 `R1` 时才处理 `<think>` 标签
- 非 R1 模型直接原样返回
- `model_source` 和 `api_model_name` / `selected_model` 是 **全局变量**，在函数外定义

### 3.2 三种正则替换

**情况 1：完整 `<think>...</think>`**

```python
re.sub(r'(<think>)(.*?)(</think>)',
       r'<details ...><summary>推理内容（展开）</summary>\2</details>',
       content, flags=re.DOTALL)
```

- `re.DOTALL`：让 `.` 匹配换行符，跨段落匹配思考内容
- 转换为 HTML `<details>` 标签，默认折叠
- `\2` 引用第二个捕获组（思考内容）

**情况 2：只有 `<think>` 没有 `</think>`（正在生成中）**

```python
re.sub(r'<think>(.*?)$',
       r'<details open><summary>推理中...</summary>\1</details>',
       content, flags=re.DOTALL)
```

- `open` 属性让 `<details>` 默认展开
- 标题变为"推理中..."

**情况 3：没有 `<think>` 但有 `</think>`**

```python
re.sub(r'(.*?)</think>',
       r'<details><summary>推理内容（展开）</summary>\1</details>',
       content, flags=re.DOTALL)
```

- 捕获 `<think>` 之前的全部内容

### 3.3 渲染时序

本地模式下，每次流式 token 到达都会调用 `process_assistant_content`：

```
第 1 次：""                                    → ""
第 2 次："<think>"                             → <details open>推理中...</details>
第 3 次："<think>我先分析一下"                    → <details open>推理中...我先分析一下</details>
...
第 N 次："<think>我先分析一下\n因此答案是42</think>"  → <details>推理内容（展开）...先分析一下\n因此答案是42</details>
第 N+1 次："最终答案是42"                         → <details>...</details>最终答案是42
```

这样用户在流式生成过程中看到的是展开的"推理中..."框，生成完成后自动折叠。

---

## 第四章：模型加载（本地模式）

```python
@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval().to(device)
    return model, tokenizer
```

### 4.1 `@st.cache_resource` 的作用

- Streamlit 的**缓存装饰器**：每次用户交互都会重新执行脚本，但加了 `@st.cache_resource` 的函数的返回值会被缓存
- 模型加载是**重量级操作**（几秒到几十秒），不加缓存的话每次打字都会重新加载
- `@st.cache_resource` 专门缓存**全局资源**（模型、数据库连接等），而 `@st.cache_data` 缓存**数据**（DataFrame 等）
- 缓存的 key 是 `model_path`——选择不同模型时自动重新加载

### 4.2 `trust_remote_code=True`

- 允许从 HuggingFace Hub 加载自定义模型代码
- MiniMind 在 HuggingFace 上注册了 `MiniMindForCausalLM`，需要此参数

### 4.3 MODEL_PATHS 结构

```python
MODEL_PATHS = {
    "MiniMind2-R1 (0.1B)":       ["../MiniMind2-R1",       "MiniMind2-R1"],
    "MiniMind2-Small-R1 (0.02B)":["../MiniMind2-Small-R1", "MiniMind2-Small-R1"],
    "MiniMind2 (0.1B)":          ["../MiniMind2",           "MiniMind2"],
    "MiniMind2-MoE (0.15B)":     ["../MiniMind2-MoE",       "MiniMind2-MoE"],
    "MiniMind2-Small (0.02B)":   ["../MiniMind2-Small",     "MiniMind2-Small"]
}
```

- key：显示在下拉框中的名称
- value[0]：模型路径（相对于 `scripts/` 的上级目录 `../`）
- value[1]：模型名称标识（用于判断是否为 R1 模型）
- 默认选择 `index=2`（MiniMind2 0.1B）

> **注意**：这些 HuggingFace 格式的模型需要通过 `scripts/Tools/convert_model.py` 转换得到，或是从 HuggingFace 下载的独立仓库。这些路径是**示例路径**，实际使用需要根据你的模型存放位置修改。

---

## 第五章：双模式生成逻辑（核心）

### 5.1 API 模式

```python
if model_source == "API":
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_url)
    history_num = st.session_state.history_chat_num + 1  # +1 包含当前消息
    conversation_history = system_prompt + st.session_state.chat_messages[-history_num:]
    response = client.chat.completions.create(
        model=api_model_id,
        messages=conversation_history,
        stream=True,
        temperature=st.session_state.temperature
    )
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        answer += content
        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)
```

这与 `chat_openai_api.py` 的流式处理完全一致：

| 组件 | chat_openai_api.py | web_demo.py |
|------|-------------------|-------------|
| 历史切片 | `[-(history_messages_num or 1):]` | `[-(history_chat_num + 1):]` |
| 流式拼接 | `chunk.choices[0].delta.content or ""` | 完全相同 |
| 显示 | `print(..., end="")` | `placeholder.markdown()` |
| 错误处理 | 无 | `try/except Exception as e` |

**关键差异**：

- `history_chat_num + 1` 加 1 包含当前刚追加的用户消息（因为切片发生在用户消息已追加之后）
- 使用 `placeholder.markdown` 实时更新界面，而不是终端打印
- 有 `try/except` 捕获异常，显示友好错误信息

### 5.2 本地模式

```python
# 1. 设置随机种子
random_seed = random.randint(0, 2 ** 32 - 1)
setup_seed(random_seed)

# 2. 切片历史并应用 chat template
st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-(history_chat_num + 1):]
new_prompt = tokenizer.apply_chat_template(
    st.session_state.chat_messages, tokenize=False, add_generation_prompt=True
)

# 3. Tokenize
inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

# 4. 创建 Streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 5. 配置生成参数
generation_kwargs = {
    "input_ids": inputs.input_ids,
    "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
    "num_return_sequences": 1,
    "do_sample": True,
    "attention_mask": inputs.attention_mask,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "temperature": st.session_state.temperature,
    "top_p": 0.85,        # 硬编码，没有通过侧边栏暴露
    "streamer": streamer,
}

# 6. 分离线程生成
Thread(target=model.generate, kwargs=generation_kwargs).start()

# 7. 主线程读取
for new_text in streamer:
    answer += new_text
    placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)
```

#### 5.2.1 `TextIteratorStreamer`

- HuggingFace Transformers 提供的**流式解码器**
- 在生成过程中逐 token 吐出文本，无需等待全部生成完成
- 内部通过 **队列（Queue）** 实现线程间通信
- `skip_prompt=True`：不输出 prompt 文本，只输出生成的 token
- `skip_special_tokens=True`：跳过特殊 token（如 `<eos>`）

#### 5.2.2 `Thread` 的分离设计

```python
Thread(target=model.generate, kwargs=generation_kwargs).start()
for new_text in streamer:
    ...
```

- `model.generate()` 是**阻塞调用**，会一直运行到生成结束或触发 EOS
- 放在子线程中运行，主线程通过 `streamer` 迭代器实时获取已生成的 token
- 这种设计让 Streamlit 能在生成过程中**实时更新 UI**（`placeholder.markdown`）

#### 5.2.3 `setup_seed` 随机种子

```python
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

- 每轮对话使用 `random.randint` 生成随机种子
- 确保即使在相同输入下每次生成结果也不同（`do_sample=True`）
- `cudnn.deterministic = True` 确保 GPU 计算的可重现性

---

## 第六章：历史会话管理

### 6.1 双列表设计

```python
st.session_state.messages = []        # 用于界面渲染
st.session_state.chat_messages = []   # 用于 API 请求/模型生成
```

为什么需要两个列表？

| 列表 | 用途 | 内容 |
|------|------|------|
| `messages` | `st.markdown` 渲染气泡 + 删除按钮 | 用户和助手的完整消息 |
| `chat_messages` | 发送给 `chat completions` 或 `apply_chat_template` | 仅 role + content |

实际上两个列表存储的内容**完全相同**（都 append 了 `{"role": "user/assistant", "content": "..."}`），这是一个冗余设计。

### 6.2 历史轮数控制

```python
st.session_state.history_chat_num = st.sidebar.slider(
    "Number of Historical Dialogues", 0, 6, 0, step=2
)
```

- 范围 0~6，步长 2，默认 0
- `step=2` 确保始终为偶数（user + assistant 成对）
- 在切片时实际使用 `history_chat_num + 1`（因为当前消息已追加）

### 6.3 删除对话

文件中有两套删除逻辑：

**方案 1：`init_chat_messages` 中的 🗑 按钮**

```python
if st.button("🗑", key=f"delete_{i}"):
    st.session_state.messages.pop(i)
    st.session_state.messages.pop(i - 1)      # 删除 user 消息
    st.session_state.chat_messages.pop(i)
    st.session_state.chat_messages.pop(i - 1)  # 删除对应的 chat_messages
    st.rerun()
```

- 按 `index, index-1` 弹出两条（user + assistant 配对）
- 双列表都有同样的操作

**方案 2：`main` 中的 × 按钮**

```python
if st.button("×", key=f"delete_{len(messages) - 1}"):
    st.session_state.messages = st.session_state.messages[:-2]
    st.session_state.chat_messages = st.session_state.chat_messages[:-2]
    st.rerun()
```

- 切片删除最后 2 条（最新一轮的 user + assistant）
- 只出现在最新一条助手消息之后

一个明显的 **Bug**：`init_chat_messages` 在 `main()` **之前**执行，但里面的删除按钮 `st.button("🗑", key=f"delete_{i}")` 因为 Streamlit 的**组件 key 唯一性要求**，会与 `main()` 中的 `st.button("×", key=f"delete_{len(messages) - 1}")` 产生 key 冲突。实际上 `init_chat_messages` 函数在文件中虽然定义了，但在 `main()` 中**从未被调用**，所以没有实际影响。

---

## 第七章：Streamlit 组件详解

### 7.1 核心组件

| 组件 | 用法 | 说明 |
|------|------|------|
| `st.chat_message` | `with st.chat_message("assistant", avatar=image_url)` | 聊天气泡容器，支持头像 |
| `st.chat_input` | `st.chat_input(key="input", placeholder="...")` | 底部输入框 |
| `st.empty` | `placeholder = st.empty()` | 占位容器，后续用 `.markdown()` 更新 |
| `st.markdown` | `st.markdown("...", unsafe_allow_html=True)` | 渲染 Markdown/HTML |
| `st.button` | `st.button("🗑", key="delete_0")` | 按钮，key 必须唯一 |
| `st.sidebar.slider` | `st.sidebar.slider("label", min, max, default, step)` | 侧边栏滑块 |
| `st.sidebar.radio` | `st.sidebar.radio("label", options, index)` | 侧边栏单选框 |
| `st.sidebar.selectbox` | `st.sidebar.selectbox("label", options, index)` | 侧边栏下拉框 |
| `st.sidebar.text_input` | `st.sidebar.text_input("label", value)` | 侧边栏文本输入 |
| `st.rerun` | `st.rerun()` | 强制重新运行脚本 |

### 7.2 `st.session_state` 的状态管理

Streamlit 是**从上到下重新执行脚本**的模型，所有跨交互的状态必须保存在 `st.session_state` 中：

```python
# 初始化（每次页面加载时执行一次）
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_messages = []
```

`st.session_state` 是**字典风格**的全局状态，在用户交互之间保持：
- 追加消息：`st.session_state.messages.append(...)`
- 删除消息：`st.session_state.messages.pop(...)`
- 清除：`del st.session_state.messages`

### 7.3 流式更新机制

```python
placeholder = st.empty()           # 创建空占位
for new_text in streamer:
    answer += new_text
    placeholder.markdown(...)       # 每次迭代更新内容
```

- `st.empty()` 创建一个"可变容器"
- 每次调用 `.markdown()` 会**替换**之前的内容，而不是追加
- 这让前端看到的是"逐字增长"的效果

---

## 第八章：与 chat_openai_api.py 的完整对比

| 维度 | chat_openai_api.py | web_demo.py |
|------|-------------------|-------------|
| **行数** | 33 行 | 328 行 |
| **定位** | 纯命令行客户端 | 图形化 Web 界面 |
| **框架** | 无（终端 I/O） | Streamlit |
| **本地模型** | 不支持 | 支持（HuggingFace 直接加载） |
| **API 模式** | 支持 | 支持 |
| **双模式** | 仅 API | 本地 + API |
| **流式展示** | `print(..., end="")` 逐字打印 | `placeholder.markdown` 实时渲染 |
| **历史控制** | 硬编码变量 | 侧边栏滑块动态调整 |
| **生成参数** | 硬编码 | 侧边栏滑块调整 |
| **错误处理** | 无 | try/except 友好提示 |
| **多模型选择** | 不支持 | 下拉框选择 |
| **删除对话** | 不支持（重启清零） | 按钮删除 |
| **system prompt** | 无 | `system_prompt = []`（空列表占位） |
| **代码复杂度** | 简单循环 | 流式渲染 + 状态管理 |
| **适用场景** | 快速测试 API | 产品化的聊天界面 |

---

## 第九章：web_demo.py 的 Bug 分析

这是本项目**代码质量较高的发现练习**，当前版本存在以下问题：

### Bug 1：CSS 样式覆盖

两段 `.stButton button` 规则冲突，`all: unset` 覆盖了之前的所有样式。

### Bug 2：`init_chat_messages` 不被调用

`init_chat_messages()`、`clear_chat_messages()`、`regenerate_answer()`、`delete_conversation()` 四个函数定义了但从未被调用。历史消息的渲染由 `main()` 中的循环独立完成。

### Bug 3：`prompt` 截断逻辑

```python
messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
```

这是**字符级别**截断，不是 token 级别。如果 `max_new_tokens=256`，只取用户输入的后 256 个字符，可能截断在中文 UTF-8 字符的中间，导致乱码。

### Bug 4：`chat_messages` 在循环中被修改

```python
st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-(history_chat_num + 1):]
```

这一行**原地修改**了 `st.session_state.chat_messages`，替换为切片后的子集。下一轮对话时，`chat_messages` 已经丢失了之前的全部历史。这是因为 `apply_chat_template` 需要完整的消息列表，所以**临时**切片传入，但同时**永久**截断了 `chat_messages`。

**正确做法**应该是保存一个完整历史列表 `chat_messages_all`，切片只在调用 template 时使用。

### Bug 5：双列表冗余

`messages` 和 `chat_messages` 存储完全相同的内容，但没有同步保证。如果其中一个被修改而另一个没改，会出现不一致。

---

## 第十章：动手练习

### 基础练习

1. **启动体验**：用本地模式启动 web_demo.py，选择不同模型聊天，观察生成速度和效果差异

2. **参数调节**：调整 History（0/2/4/6）、Temperature（0.6/0.85/1.2）、Max Length（256/2048/8192），观察生成效果变化

3. **API 模式对比**：先启动 `serve_openai_api.py`，再用 API 模式连接，对比同一模型在本地模式和 API 模式下的差异

4. **猜测 Bug**：运行 web_demo.py，输入几轮对话后观察 history 设置是否按预期工作（特别是 `history_chat_num > 0` 时），验证 Bug 4 是否存在

### 进阶练习

5. **修复 Bug 4（chat_messages 历史丢失）**：修改代码，使用 `chat_messages_all` 保存完整历史，`chat_messages[-n:]` 仅在 template 时切片

6. **将 `top_p` 加入侧边栏**：当前 `top_p=0.85` 是硬编码的，把它改为侧边栏滑块（`st.sidebar.slider("Top P", 0.1, 1.0, 0.85, step=0.05)`）

7. **支持非流式模式**：在侧边栏加一个 `st.sidebar.checkbox("流式输出", value=True)` 勾选框，未勾选时一次性显示完整回复

8. **在界面显示 token 用量**：非流式模式下，在回复底部显示生成 token 数和耗时

9. **修复 Bug 3（字符级截断）**：将 `prompt[-max_new_tokens:]` 改为 token 级截断（先 encode 再 decode slice）

### 深入练习

10. **增加 `--server.port` 参数**：`streamlit run` 支持 `--server.port`，但也可以在代码中通过 `st.set_option` 设置。添加一个 `if __name__ == "__main__"` 入口来解析 argparse 参数

11. **自定义 CSS 美化**：修改 CSS，给对话气泡添加渐变背景、动画效果或暗色主题

12. **多轮对话 token 用量统计**：在侧边栏显示当前会话的总 token 数、prompt token 数（作为 tokenizer 使用的练习）

13. **增加重新生成功能**：调用 `regenerate_answer` 函数（当前已定义但未使用），在每条助手消息旁加一个"🔄"按钮

---

## 自测题

1. **`@st.cache_resource` 和 `@st.cache_data` 有什么区别？为什么模型加载用前者？**

2. **`st.session_state` 为什么能跨用户交互保持状态？它的底层实现是什么？**

3. **为什么 `model.generate()` 需要放在 `Thread` 中执行？如果不放子线程会怎样？**

4. **`TextIteratorStreamer` 的内部机制是什么？`skip_prompt` 和 `skip_special_tokens` 各控制什么？**

5. **`process_assistant_content` 中三个正则替换分别处理什么情况？为什么需要 `re.DOTALL`？**

6. **对比本地模式和 API 模式，各有什么优缺点？什么时候用哪个？**

7. **`st.empty().markdown()` 如何实现逐字更新效果？多次调用 `markdown` 是追加还是替换？**

8. **为什么 `history_chat_num` 要设置为 `step=2`？设为奇数会怎样？**

9. **本文件中哪些函数定义了但从未被调用？这对程序行为有影响吗？**

10. **`st.session_state.chat_messages[-(history_chat_num + 1):]` 中为什么要 `+1`？**

11. **`MODEL_PATHS` 中的路径为什么是 `../MiniMind2` 而不是绝对路径？如果从项目根目录运行 `streamlit run` 会怎样？**

12. **对比 `eval_llm.py` 的 `enable_thinking` 参数和 `web_demo.py` 的 `process_assistant_content`，两者处理思考内容的思路有什么不同？**

---

## 自测题参考答案

<details>
<summary>点击展开参考答案</summary>

### Q1: `@st.cache_resource` vs `@st.cache_data`

- `@st.cache_resource`：缓存**全局资源对象**（模型、数据库连接、HTTP 客户端），返回的对象是**共享的**，不可变
- `@st.cache_data`：缓存**数据**（DataFrame、numpy 数组、字符串），使用 pickle 序列化
- 模型加载用 `@st.cache_resource` 因为模型是重量级、不可序列化的对象，而且所有用户/交互应该共享同一个模型实例

### Q2: `st.session_state` 的实现

Streamlit 在每次交互时重新执行整个脚本，但 `st.session_state` 是一个基于**前端浏览器**的 JSON 状态对象。每次重新运行时，前端会把 session_state 作为 WebSocket 消息的一部分发送给后端，后端反序列化后注入到 Python 环境中。修改后，Python 又序列化回前端存储。这是一种**前后端同步的状态管理**机制。

### Q3: 为什么需要 Thread

`model.generate()` 是一个同步阻塞调用，在生成完成之前不会返回。如果不放子线程：
- `for new_text in streamer` 循环不会开始（streamer 的迭代器需要等 generate 开始才能输出 token）
- UI 会完全卡住，直到全部生成完毕才一次性显示
- Streamlit 会认为脚本执行完毕，用户看到的是"全部出现"而不是"逐字出现"

### Q4: TextIteratorStreamer 的内部机制

`TextIteratorStreamer` 继承自 `TextStreamer`，内部包含一个 `Queue`。`model.generate()` 在生成每个 token 时回调 `on_finalized_text`，把解码后的文本放入队列。主线程通过 `for new_text in streamer:` 从队列中阻塞读取，实现了**生产者-消费者模式**。

- `skip_prompt=True`：不输出 prompt 文本（只输出新生成的 token）
- `skip_special_tokens=True`：过滤掉特殊 token（如 `<eos>`、`<pad>`）

### Q5: 三种正则替换

见第三章 3.2 节。`re.DOTALL` 让 `.` 匹配换行符，因为思考内容通常是跨多行的完整段落。

### Q6: 本地模式 vs API 模式

| 模式 | 优点 | 缺点 |
|------|------|------|
| 本地 | 无需额外服务、延迟低、可离线使用 | 占用 GPU 显存、模型切换需要重新加载 |
| API | 可远程访问、支持多客户端共享、负载分离 | 需要先启动服务、网络延迟、增加复杂度 |

### Q7: `st.empty().markdown()` 的更新行为

每次调用 `.markdown()` **替换**之前的内容，不是追加。这实现了"原地更新"的效果，让用户看到文字逐字增长。

### Q8: `step=2` 的原因

`history_chat_num` 表示历史对话轮数，每轮包含 user + assistant 两条消息。步长为 2 确保只能选择偶数，避免出现只有 user 没有 assistant 或反之的奇数情况。

### Q9: 未被调用的函数

`init_chat_messages()`、`clear_chat_messages()`、`regenerate_answer()`、`delete_conversation()` 四个函数定义了但从未被调用。不影响程序行为，但属于死代码，在重构时应该清理或补全调用逻辑。

### Q10: `+1` 的原因

`history_chat_num` 表示历史完整轮数。当用户输入新消息后，该消息已先被追加到 `chat_messages`。切片时 `+1` 是为了包含**当前这条刚追加的用户消息**。例如 `history_chat_num=2`（表示携带最近 2 轮历史），实际切片取 `[-(2+1):]` = 最后 3 条 = 上一轮 assistant + 上一轮 user + 当前 user。

### Q11: 路径问题

如果从项目根目录运行 `streamlit run scripts/Deploy/web_demo.py`，脚本的 `__file__` 是 `scripts/Deploy/web_demo.py`，但 `MODEL_PATHS` 中的 `../MiniMind2` 是相对于**运行目录**（项目根目录）解析的，实际上是 `./MiniMind2`。而`TextIteratorStreamer` 的 `from_pretrained` 使用相对路径时，也是相对于 CWD。所以如果 `../MiniMind2` 目录存不存在取决于模型是否已下载到项目根目录。

> 这种硬编码的相对路径是设计上的缺陷——最佳实践应该是用命令行参数或环境变量指定模型路径。

### Q12: eval_llm.py 的 enable_thinking vs web_demo.py 的 process_assistant_content

- `eval_llm.py`：在**生成之前**通过 `enable_thinking` 参数控制模型是否输出 `<think>` 标记，并且在生成后通过 `split("<think>")` 做**简单字符串分割**，用 `print` 分两次输出
- `web_demo.py`：不控制模型是否输出 `<think>`，而是在**生成之后**通过**正则替换**把标签渲染为 HTML `<details>` 标签，支持折叠展开，用户体验更好
- 本质区别：eval_llm.py 是**预处理控制**（让模型决定要不要思考），web_demo.py 是**后处理渲染**（无论输出什么都能美观展示）

</details>

---

## 拓展阅读

- [Streamlit 官方文档](https://docs.streamlit.io/)
- [Streamlit session_state 指南](https://docs.streamlit.io/library/advanced-features/session-state)
- [HuggingFace TextStreamer API](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextStreamer)
- [serve_openai_api.py 学习计划](./serve_openai_api_study_plan.md) — API 服务端
- [chat_openai_api.py 学习计划](./chat_openai_api_study_plan.md) — API 客户端对照
