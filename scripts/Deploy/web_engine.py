"""web_demo 的生成引擎：本地模型（含跨轮前缀缓存）与 OpenAI 兼容 API。不依赖 streamlit。"""
from dataclasses import dataclass
from threading import Thread

import torch

from scripts.Deploy.kv_generate import SamplingParams, generate_hf_cache, generate_kv


@dataclass
class LocalParams:
    """本地模型生成参数。hf_model：True=HF 格式模型（DynamicCache），False=原生 .pth（legacy 元组缓存）。"""
    max_new_tokens: int = 8192
    temperature: float = 0.85
    top_p: float = 0.85
    device: str = 'cpu'
    use_kv: bool = False
    hf_model: bool = True


def local_generate(model, tokenizer, messages, params: LocalParams, kv, streamer):
    """启动本地模型生成线程（经 streamer 输出），返回 holder 供取回更新后的 cache。

    kv: {"cache": ..., "all_ids": ...}；use_kv=True 时跨轮维护前缀缓存
    （全量渲染 + 前缀求差，只 prefill 增量；前缀不匹配自动回退全量重算）。
    """
    holder = {"cache": None, "thread": None}
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if params.use_kv:
        full_ids = tokenizer(new_prompt, truncation=True).input_ids
        cached = kv["all_ids"][0].tolist() if kv["all_ids"] is not None else None
        if cached is not None and full_ids[:len(cached)] == cached:
            chunk = torch.tensor(full_ids[len(cached):], dtype=torch.long, device=params.device).unsqueeze(0)
            cache = kv["cache"]
        else:
            cache = None
            chunk = torch.tensor(full_ids, dtype=torch.long, device=params.device).unsqueeze(0)
        kv["all_ids"] = torch.tensor(full_ids, dtype=torch.long, device=params.device).unsqueeze(0)
        streamer.put(chunk)  # 解锁 skip_prompt：首个 put 会被吞掉，之后生成的 token 才会显示

        if params.hf_model:
            def _run_hf():
                _, holder["cache"] = generate_hf_cache(
                    model, chunk, cache,
                    max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )
            holder["thread"] = Thread(target=_run_hf)
            holder["thread"].start()
        else:
            def _run_native():
                _, holder["cache"], kv["all_ids"] = generate_kv(
                    model, chunk, kv["all_ids"], cache,
                    eos_token_id=tokenizer.eos_token_id,
                    params=SamplingParams(
                        max_new_tokens=params.max_new_tokens,
                        temperature=params.temperature,
                        top_p=params.top_p,
                    ),
                    streamer=streamer,
                )
            holder["thread"] = Thread(target=_run_native)
            holder["thread"].start()
    else:
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(params.device)
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + params.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "streamer": streamer,
        }
        holder["thread"] = Thread(target=model.generate, kwargs=generation_kwargs)
        holder["thread"].start()

    return holder


def api_generate(client, model_id, messages, temperature):
    """流式调用 OpenAI 兼容 API，逐段 yield 累积的回复文本。"""
    answer = ""
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        temperature=temperature,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        answer += content
        yield answer
