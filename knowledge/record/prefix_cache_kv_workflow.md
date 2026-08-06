# MiniMind 多轮 KV / Prefix Cache 工作流

> 本文档用纯文本讲解「客户端多轮 + 服务端 prefix caching」的完整工作流（不依赖 mermaid 渲染）。
> 配套排障记录见 `knowledge/record/prefix_cache_kv_impl.md`。
> 涉及文件：`chat_openai_api.py`（客户端）、`serve_openai_api.py --enable_kv`（服务端）、`generate_kv.py`（解码循环）、`prefix_cache.py`（前缀缓存）。

---

## 一、三个角色

| 角色 | 文件 | 职责 |
| --- | --- | --- |
| **客户端** | `chat_openai_api.py` | 每轮把**完整对话历史**发给服务端 |
| **服务端** | `serve_openai_api.py --enable_kv` | 接收请求 → 找前缀 → 只算增量 → 返回回答 |
| **模型** | `MiniMindForCausalLM` + `generate_kv` | 逐 token 解码，维护 K/V |

要点：客户端只管发完整历史，服务端管缓存与复用，模型管计算。

---

## 二、一次请求在服务端内部的 5 步

```
请求进来（messages = 完整历史）
   │
   ① tokenize：整段对话渲染成模板文本 → 一串 token（full_ids）
   │
   ② find_prefix：拿这串 token 去缓存里找"最长能对上的前缀"
   │       ┌─ 命中：只取前缀后面的"新 token"（增量）→ 复用缓存的 past_key_values
   │       └─ 未命中：整段都算（全量 prefill），past_key_values = None
   │
   ③ generate_kv：只把增量喂给模型，逐 token 生成回复
   │
   ④ store_prefix：把"最新完整 token 流 + 新 K/V"存回缓存
   │
   ⑤ 返回回复（流式或非流式）
```

对应代码：

```python
full_ids = tokenizer(...).input_ids[0].tolist()            # ① 整段 token
prefix_len, past_key_values = find_prefix(full_ids)        # ② 找最长前缀
chunk_ids = full_ids[prefix_len:]                          # ③ 只取增量
generated_ids, past_key_values, all_new = generate_kv(...) # ③ 解码
store_prefix(all_new, past_key_values)                     # ④ 存回缓存
```

---

## 三、三轮对话的缓存变化（核心）

| 轮次 | 客户端发送 | tokenize 总长 | 前缀命中 | 实际只算了 | 回复后缓存流长度 |
| --- | --- | --- | --- | --- | --- |
| 第 1 轮 | `[user: 你好]` | 25 | 0/25（冷启动） | 25（全量） | 53 |
| 第 2 轮 | `[user, assistant, user: 你叫什么]` | 52 | **34/52** | 18 | 77 |
| 第 3 轮 | `[user, assistant, user, assistant, user]` | 98 | **61/77** | 16 | 100 |

解读：

- **第 1 轮**：缓存为空，25 个 token 全算，生成后缓存里是 53 个 token 的 K/V（25 prompt + 28 回复）。
- **第 2 轮**：完整历史 tokenize 出 52 个，与缓存前缀对上 34 个（系统提示 + 你好 + 回复1 + 结束符），只把后面 18 个新 token 喂给模型，前面 34 个 K/V 直接从缓存拿。
- **第 3 轮**：缓存涨到 77，命中前 61 个，只算 16 个新的。

本质：每轮只算"新增的那几行"，前面所有历史的 K/V 都是上次算好存起来的。

---

## 四、为什么"前缀"能对上（方案成立的前提）

每轮请求进来，服务端都用 `apply_chat_template` 把**完整历史重新渲染一遍**再 tokenize；而上一轮结束时，缓存里存的是"上一轮 prompt + 上一轮回复"的完整 token 流。

因为客户端把上一轮的回复**原样**存回历史：

```
第 2 轮的完整渲染 = 第 1 轮的 prompt + 第 1 轮的回复 + 第 2 轮的新问题
缓存里存的东西   =  第 1 轮的 prompt + 第 1 轮的回复     ← 正好是上面那串的前缀
```

两者从头逐 token 一致（已验证 re-encode 无损），于是可以复用。

**如果对不上**（历史被裁剪 / 文本往返有差异）→ `find_prefix` 返回 0 → 自动回退全量重算。**结果永远正确，只是这一轮吃不到加速**。

---

## 五、和"每轮全量重算"的对比

| | 老方案（无缓存） | 新方案（--enable_kv） |
| --- | --- | --- |
| 第 2 轮计算量 | 重新算 52 个 token | 只算 18 个 |
| 第 N 轮计算量 | 每轮重算整段历史（累计 O(N²)） | 每轮只算新增块（累计 O(N)） |
| 10 轮累计 | 约 2430 token 的 prefill | 约 180 token 的 prefill |
| 代价 | 省显存 | 缓存 K/V 占显存，随对话增长 |
| 结果 | — | 与全量重算完全等价（前缀对不上会回退） |

---

## 六、一句话总结

> **客户端每轮发完整历史 → 服务端把历史 tokenize 后去缓存里找最长前缀 → 只把前缀后面的新增 token 喂给模型 → 模型用缓存的 K/V 续算并生成回复 → 把新的完整 token 流和 K/V 存回缓存，供下一轮复用。**

也就是把「每轮重算所有历史」变成「每轮只算新增的几行，其余直接从缓存里取」。
