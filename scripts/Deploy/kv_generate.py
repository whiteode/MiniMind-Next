"""逐 token 自回归解码与采样逻辑。

- generate_kv：跨轮维护 past_key_values（legacy 格式 List[Tuple[k, v]]）的生成循环
- apply_sampling / sample_token：纯采样函数，便于单元测试
- SamplingParams：采样与生成长度参数集合
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    """采样与生成长度参数。"""
    max_new_tokens: int = 8192
    temperature: float = 0.85
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 1.0


def apply_sampling(logits, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, seen=None):
    """对 logits 应用 temperature / repetition_penalty / top_k / top_p 过滤，返回过滤后的 logits。
    seen：已生成 token 序列（repetition_penalty 用，可为 None）。"""
    logits = logits / temperature
    if repetition_penalty != 1.0 and seen is not None:
        seen = torch.unique(seen)
        score = logits[0, seen]
        logits[0, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float('-inf')), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        logits = logits.masked_fill(mask.scatter(-1, sorted_indices, mask), float('-inf'))
    return logits


def sample_token(logits, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, seen=None):
    """对 logits 采样，返回 [1, 1] 的 next token。"""
    logits = apply_sampling(logits, temperature=temperature, top_p=top_p, top_k=top_k,
                            repetition_penalty=repetition_penalty, seen=seen)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.no_grad()
def generate_kv(model, input_ids, all_ids, past_key_values, eos_token_id,
                params: Optional[SamplingParams] = None, streamer=None):
    """逐 token 自回归解码，跨轮维护 past_key_values（每层一个 (k, v) 元组列表）。
    返回 (生成token张量, 更新后的cache, 累计token张量)。"""
    params = params or SamplingParams()
    generated = []
    for _ in range(params.max_new_tokens):
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        next_token = sample_token(
            outputs.logits[:, -1, :],
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            seen=all_ids,
        )
        generated.append(next_token)
        all_ids = torch.cat([all_ids, next_token], dim=1)
        input_ids = next_token
        past_key_values = outputs.past_key_values
        if streamer is not None:
            streamer.put(next_token)
        if next_token.item() == eos_token_id:
            break

    if streamer is not None:
        streamer.end()

    if not generated:
        return torch.empty(1, 0, dtype=torch.long, device=input_ids.device), past_key_values, all_ids
    return torch.cat(generated, dim=1), past_key_values, all_ids
