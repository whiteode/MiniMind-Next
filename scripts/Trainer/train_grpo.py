import os
import sys

sys.path.insert(0, os.getcwd())

import argparse
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.Dataset.lm_dataset import RLAIFDataset
from scripts.Trainer.train_common import (
    add_train_args, init_train_env, make_lm_config, make_autocast_ctx, init_wandb,
    get_ckp_data, load_resume_state, wrap_ddp, for_each_epoch, save_checkpoint,
    calculate_rewards,
)
from scripts.Trainer.trainer_utils import Logger, is_main_process, init_model

warnings.filterwarnings('ignore')


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        with torch.no_grad():
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)

        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]

        def get_per_token_logps(mdl, input_ids, n_keep):
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)

        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer,
                                    reasoning=(args.reasoning == 1), device=args.device,
                                    num_generations=args.num_generations)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config, model, optimizer, args, epoch, step,
                            scheduler=scheduler, wandb=wandb)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="internlm2-1_8b-reward", help="Reward模型路径")
    add_train_args(parser, save_weight='grpo', epochs=1, batch_size=2,
                   learning_rate=8e-8, log_interval=1, save_interval=10,
                   max_seq_len=66, data_path='resource/minimind_dataset/rlaif.jsonl',
                   wandb_project='MiniMind-GRPO', from_weight=None)
    args = parser.parse_args()

    local_rank = init_train_env(args)
    lm_config = make_lm_config(args, max_seq_len=args.max_seq_len + args.max_gen_len)
    ckp_data = get_ckp_data(args, lm_config)
    autocast_ctx = make_autocast_ctx(args)
    wandb = init_wandb(args, f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}", ckp_data)

    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    total_optimizer_steps = (len(loader_for_count) // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scheduler=scheduler)
    model = wrap_ddp(model, local_rank)

    for epoch, loader, iters, st, _ in for_each_epoch(
            train_ds, args, start_epoch, start_step, train_sampler=train_sampler):
        grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, st, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
