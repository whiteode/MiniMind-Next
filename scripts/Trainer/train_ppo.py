import os
import sys

sys.path.insert(0, os.getcwd())

import argparse
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.Model.model_minimind import MiniMindForCausalLM
from scripts.Dataset.lm_dataset import RLAIFDataset
from scripts.Trainer.train_common import (
    add_train_args, init_train_env, make_lm_config, make_autocast_ctx, init_wandb,
    get_ckp_data, load_resume_state, wrap_ddp, for_each_epoch, save_checkpoint,
    calculate_rewards,
)
from scripts.Trainer.trainer_utils import Logger, is_main_process, init_model

warnings.filterwarnings('ignore')


class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len, padding_side="left").to(args.device)
        prompt_length = enc.input_ids.shape[1]

        with torch.no_grad():
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

        responses_text = [tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer,
                                    reasoning=(args.reasoning == 1), device=args.device)

        full_mask = (gen_out != tokenizer.pad_token_id).long()
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]
        advantages = rewards - values.detach()

        with autocast_ctx:
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        
        labels = gen_out[:, 1:].clone()
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))
        actor_logp = (logp_tokens * final_mask).sum(dim=1)

        with torch.no_grad():
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)
            
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)

        kl = (actor_logp - old_logp).mean()
        kl_ref = (actor_logp - ref_logp).mean()
        ratio = torch.exp(actor_logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, rewards)
        loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if is_main_process():
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                   f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            state_dict = raw_actor.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config, actor_model, actor_optimizer, args, epoch, step,
                            scheduler=actor_scheduler, wandb=wandb,
                            extra_state={'critic_model': critic_model,
                                         'critic_optimizer': critic_optimizer,
                                         'critic_scheduler': critic_scheduler})

        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--critic_learning_rate", type=float, default=1e-6, help="Critic学习率")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="internlm2-1_8b-reward", help="Reward模型路径")
    add_train_args(parser, save_weight='ppo_actor', epochs=1, batch_size=2,
                   learning_rate=1e-6, log_interval=1, save_interval=10,
                   max_seq_len=66, data_path='resource/minimind_dataset/rlaif.jsonl',
                   wandb_project='MiniMind-PPO', from_weight=None)
    args = parser.parse_args()

    local_rank = init_train_env(args)
    lm_config = make_lm_config(args)
    ckp_data = get_ckp_data(args, lm_config)
    autocast_ctx = make_autocast_ctx(args)
    wandb = init_wandb(args, f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}", ckp_data)

    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(
        torch.load(f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth',
                   map_location=args.device),
        strict=False,
    )
    critic_model = critic_model.to(args.device)
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    total_optimizer_steps = (len(loader_for_count) // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    start_epoch, start_step = load_resume_state(
        ckp_data, actor_model, actor_optimizer, scheduler=actor_scheduler,
        extra=[('critic_model', critic_model), ('critic_optimizer', critic_optimizer),
               ('critic_scheduler', critic_scheduler)])
    actor_model = wrap_ddp(actor_model, local_rank)
    critic_model = wrap_ddp(critic_model, local_rank)
    if dist.is_initialized():
        old_actor_model.to(args.device)

    for epoch, loader, iters, st, _ in for_each_epoch(
            train_ds, args, start_epoch, start_step, train_sampler=train_sampler):
        ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model,
                        actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, st, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
