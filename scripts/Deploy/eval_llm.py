import os
import sys
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

sys.path.insert(0, os.getcwd())
from scripts.Model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.Model.model_lora import *
from scripts.Trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    if 'model' in args.load_from.lower():
        config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        )
        model = MiniMindForCausalLM(config)
        
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        model.load_state_dict(torch.load(ckp, map_location=args.device, weights_only=False), strict=True)
        
        if args.lora_weight != 'None':
            lora_names = [w.strip() for w in args.lora_weight.split(',')]
            if len(lora_names) == 1:
                apply_lora(model)
                load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
            else:
                from scripts.Model.model_lora import apply_lora_multi, load_lora_multi
                apply_lora_multi(model, ranks=[8] * len(lora_names))
                paths = [f'./{args.save_dir}/lora/{name}_{args.hidden_size}.pth' for name in lora_names]
                load_lora_multi(model, paths)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        
    get_model_params(model, model.config)
    
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='scripts/Model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help=(
        "权重名称前缀，用于指定加载哪一阶段训练出的模型权重。"
        "各选项含义：\n"
        "  pretrain   - 预训练阶段（在海量无标注文本上学习语言建模，得到基础语言能力）\n"
        "  full_sft   - 全量指令微调（在指令数据集上全参数微调，对齐指令遵循能力，默认值）\n"
        "  rlhf       - RLHF强化学习：基于人类反馈的强化学习（Reinforcement Learning from Human Feedback），\n"
        "               核心流程分为三步：① 使用人类标注的偏好数据训练一个奖励模型（Reward Model）；\n"
        "               ② 以奖励模型给出的分数作为激励信号，对 SFT 模型进行 PPO 强化学习；\n"
        "               ③ 在最大化奖励的同时加入 KL 散度惩罚项，防止模型偏离 SFT 分布过远\n"
        "               从而在遵循人类偏好（有用性、无害性）与保持生成多样性之间取得平衡\n"
        "  reason     - 推理微调（针对数学、逻辑等推理任务进行专项微调）\n"
        "  ppo_actor  - PPO策略网络：Proximal Policy Optimization 中的 Actor 网络权重。\n"
        "               Actor 网络负责根据当前状态输出动作分布（即下一个 token 的概率分布），\n"
        "               Critic 网络评估状态价值并计算 Advantage 函数，\n"
        "               PPO 通过裁剪（clip）策略更新幅度来保证训练的稳定性，\n"
        "               此权重即 RLHF 阶段中 PPO 训练完成后 Actor 网络的参数快照\n"
        "  grpo       - GRPO策略优化：Group Relative Policy Optimization，对 PPO 的一种改进变体。\n"
        "               核心思想：对同一个 prompt 采样多个回复构成一个 group，\n"
        "               以组内回复的相对优势（而非绝对奖励模型）作为优化信号，\n"
        "               从而消除对独立 Critic 价值网络的依赖，降低训练开销\n"
        "  spo        - SPO偏好优化：Safe Policy Optimization，在偏好优化中引入安全约束。\n"
        "               相较于 DPO 仅关注偏好对齐，SPO 额外约束模型在敏感话题上的输出，\n"
        "               通过拉格朗日乘子法在奖励最大化与安全约束之间动态权衡，\n"
        "               确保模型既对齐偏好又满足安全性要求\n"
        "\n"
        "【这么多选项在 eval_llm.py 里却只用了 reason 和 pretrain 两个判断，那其他选项的意义在哪？】\n"
        "  --weight 最核心的作用不是控制 if 分支，而是定位权重文件（init_model 第 36 行）：\n"
        "    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'\n"
        "    例如 --weight full_sft --hidden_size 512  → 加载 ./out/full_sft_512.pth\n"
        "    例如 --weight rlhf --hidden_size 768      → 加载 ./out/rlhf_768.pth\n"
        "    例如 --weight reason --hidden_size 640 --use_moe 1  → 加载 ./out/reason_640_moe.pth\n"
        "  所有选项对应的权重文件都可以通过训练脚本生成（train_pretrain.py / train_full_sft.py / 等），\n"
        "  推理时用 --weight 指定加载哪个，eval_llm.py 只额外区分了 pretrain（不走 chat template）\n"
        "  和 reason（标记 enable_thinking=True）这两个需要调整行为逻辑的阶段。\n"
        "  其他阶段（rlhf / ppo_actor / grpo / spo）的加载和推理流程与 full_sft 完全一致——\n"
        "  因为 eval_llm.py 只控制\u201c加载哪个文件\u201d和\u201c对话模板是否特殊处理\u201d。\n"
        "  模型结构是同一个 MiniMindForCausalLM，不会因为训练阶段改变，\n"
        "  init_model() 里所有非 pretrain 权重走的都是同一套 load_state_dict + eval()，\n"
        "  chat_template 的渲染逻辑也完全一样（只有 reason 多传了一个 enable_thinking=True）。\n"
        "  至于 rlhf / ppo_actor / grpo / spo 这些阶段之间的差异——\n"
        "  它们是在训练策略（奖励模型、PPO clip、组相对优势、安全约束）上不同，\n"
        "  但这些差异只存于训练脚本里（train_ppo.py / train_grpo.py / train_spo.py），\n"
        "  训练完成后产出的权重在结构上和 full_sft 权重完全兼容，推理时无需区分。"
    ))
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用。支持多个LoRA用逗号分隔，如：lora_identity,lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help=(
        "生成温度，控制模型输出随机性。\n"
        "原理：模型最后一层 Linear 输出每个 token 的 logit（未归一化的分数），\n"
        "      softmax 内部先对所有 logit 除以 temperature 再做指数归一化：\n"
        "        p_i = exp(logit_i / T) / Σ_j exp(logit_j / T)\n"
        "     T → 0：logit 被放大，softmax 趋近于 one-hot（确定性 greedy 解码）\n"
        "     T → ∞：概率分布趋近于均匀分布（完全随机）\n"
        "     T ∈ (0,1]：压低低概率 token 的权重，输出更集中\n"
        "     T > 1 ：拉平分布，低概率 token 有更多出场机会，输出更多样化\n"
        "【关键】softmax 之后概率最高的 token 仍然是同一个（argmax 不变），\n"
        "      但 miniMind 用的是采样解码（do_sample=True），不会直接取 argmax，\n"
        "      而是按照 softmax 输出的概率分布去随机抽取 token。\n"
        "      T 越大 → 低概率 token 被抽中的概率越高 → 输出多样性越大；\n"
        "      T 越小 → 高概率 token 的采样优势越明显 → 输出越稳定确定。\n"
"      所以 temperature 改变的不是 argmax 是谁，而是采样时要不要选它。\n"
         "【采样解码 vs Argmax（贪心）解码】\n"
         "  Argmax 解码（do_sample=False）：每次都选概率最高的 token，输出确定但单调重复\n"
         "  采样解码（do_sample=True）：以 softmax 概率分布为权重随机抽取 token，\n"
         "    概率高的 token 被抽中的机会大，概率低的也有机会但不经常\n"
         "【那 softmax 概率还有什么用？】\n"
         "  采样并不是从所有 token 中均匀随机抽取，而是带权重的随机——\n"
         "  softmax 输出的概率就是每个 token 被抽中的权重。\n"
         "  比如某个 token 的 softmax 输出是 0.9，另一个是 0.1，\n"
         "  采样时第一个 token 被抽中的概率是 90%，第二个是 10%。\n"
         "  所以 softmax 概率分布的形状直接决定了采样的倾向性，\n"
         "  temperature 就是通过拉伸/压缩这个分布来调节倾向性强弱的。\n"
         "MiniMind 默认 0.85 略小于 1，在多样性与稳定性之间取折中"
    ))
    parser.add_argument('--top_p', default=0.85, type=float, help=(
        "nucleus（核）采样阈值，取值范围 0~1。\n"
        "作用：从采样候选集中动态裁掉尾部低概率 token，只保留累积概率达到 top_p 的核心 token。\n"
        "流程：把所有 token 按 softmax 概率从高到低排序，\n"
        "      从最高的开始累加，直到累积概率 >= top_p，\n"
        "      丢弃剩余的低概率 token，然后仅在保留的 token 中重新归一化并采样。\n"
        "例子：top_p=0.85，模型在猜下一个词时：\n"
        "       \"我喜欢吃\" → 苹果(0.6) 香蕉(0.2) 西瓜(0.1) 桌子(0.02) 月亮(0.01) ...\n"
        "       排序累加：苹果0.6 + 香蕉0.2 + 西瓜0.1 = 0.9 >= 0.85\n"
        "       所以只保留 {苹果, 香蕉, 西瓜} 三个候选，丢弃桌子、月亮等无关 token。\n"
        "作用：消除长尾里那些概率低但语义不相关的 token 被意外采到的可能，\n"
        "      让输出更干净，同时保留一定的多样性。\n"
        "配合关系：temperature 先拉伸/压缩 logit → softmax 出概率 → top_p 再截断尾部"
    ))
    parser.add_argument('--historys', default=0, type=int, help=(
        "携带历史对话轮数（需为偶数，0表示不携带历史）。\n"
        "实现方式：每次生成前用 Python 切片 conversation[-historys:] 截取列表末尾，\n"
        "         相当于一个固定大小的滑动窗口，窗口外的旧对话被丢弃，不会继续累积。\n"
        "         比如 historys=2，每轮只保留最近 2 条消息（1 轮 user+assistant）。\n"
        "         非 0 时必须为偶数，否则 user/assistant 配对会错位。"
    ))
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help="重复惩罚系数，>=1.0。1.0=不惩罚，1.1=轻微压制已出现token，2.0=强惩罚（几乎不会重复）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食',
    ]
    
    if args.weight == 'reason':
        prompts += [
            '小明有5个苹果，给了小红2个，又买了3个，现在小明有几个苹果？',
            '一个三角形两边长分别为3和4，求第三边的长度（已知是直角三角形）',
        ]
    elif args.lora_weight != 'None':
        if 'medical' in args.lora_weight:
            prompts += [
                '感冒和流感有什么区别？',
                '如何预防高血压？',
            ]
        elif 'identity' in args.lora_weight:
            prompts += [
                '请介绍一下你自己',
            ]
    
    conversation = []
    
    model, tokenizer = init_model(args)
    
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    for prompt in prompt_iter:
        setup_seed(2026) 
        
        if input_mode == 0: 
            print(f'💬: {prompt}')
            
        conversation = conversation[-args.historys:] if args.historys else []
        
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        
        if args.weight == 'reason': 
            templates["enable_thinking"] = True
            
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        
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
            repetition_penalty=args.repetition_penalty,
        )
        
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        conversation.append({"role": "assistant", "content": response})
        
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n')
        else:
            print('\n\n')

if __name__ == "__main__":
    main()
