import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_kv(model, input_ids, all_ids, past_key_values, max_new_tokens,
                temperature, top_p, repetition_penalty, eos_token_id,
                streamer=None, top_k=50):
    """逐 token 自回归解码，跨轮维护 past_key_values（每层一个 (k, v) 元组列表）。
    返回 (生成token张量, 更新后的cache, 累计token张量)。"""
    generated = []
    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :] / temperature

        if repetition_penalty != 1.0:
            seen = torch.unique(all_ids)
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

        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
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
