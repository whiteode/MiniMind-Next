import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params

# 忽略代码运行过程中的警告信息（让终端输出更干净）
warnings.filterwarnings('ignore')

def init_model(args):
    """
    初始化模型和分词器（Tokenizer）
    根据参数判断是加载原生的 MiniMind 模型还是 Hugging Face 格式的模型，并处理 LoRA 权重。
    """
    # 从指定路径加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    # 路径中包含 'model'，说明需要加载自定义的原生 PyTorch 模型结构与权重
    if 'model' in args.load_from:
        # 1. 根据传入的参数实例化 MiniMind 的配置对象
        config = MiniMindConfig(
            hidden_size=args.hidden_size,                 # 隐藏层维度
            num_hidden_layers=args.num_hidden_layers,     # Transformer 层数
            use_moe=bool(args.use_moe),                   # 是否启用 MoE (混合专家架构)
            inference_rope_scaling=args.inference_rope_scaling # 是否开启 RoPE 位置编码外推
        )
        # 2. 根据配置初始化模型结构
        model = MiniMindForCausalLM(config)
        
        # 3. 拼接权重文件的完整路径（例如: ./out/full_sft_512.pth 或 ./out/full_sft_640_moe.pth）
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        # 4. 加载 state_dict 并注入到模型中，strict=True 要求结构与权重完全匹配
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        
        # 5. 如果指定了 LoRA 权重，则动态为模型注入 LoRA 层并加载对应的 LoRA 权重
        if args.lora_weight != 'None':
            apply_lora(model) # 在模型中注入 LoRA 的旁路参数结构
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth') # 加载 LoRA 权重
    else:
        # 如果路径里不包含 'model'，则视其为标准的 Hugging Face 格式，直接通过 transformers 库加载
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        
    # 打印/计算当前模型的总参数量
    get_model_params(model, model.config)
    
    # 将模型设置为评估模式（推理模式），并移动到指定的设备（GPU/CPU）上
    return model.eval().to(args.device), tokenizer

def main():
    # 配置命令行参数解析器
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    # 预设的自动化测试 Prompt 列表
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    # 用于存储对话历史的列表，格式为 [{"role": "user", "content": "..."}, ...]
    conversation = []
    
    # 初始化模型和分词器
    model, tokenizer = init_model(args)
    
    # 引导用户选择交互模式：0 为系统预设测试，1 为终端手动打字对话
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    
    # 初始化流式传输器，实现打字机流式输出效果（跳过 Prompt 和特殊 Token）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 根据用户选择确定 Prompt 迭代器：
    # 0 模式遍历 prompts 列表；1 模式通过 iter 配合 lambda 持续获取键盘输入，直到输入为空白时停止
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    # 开始循环对话
    for prompt in prompt_iter:
        # 设置随机种子，保证每次生成结果的确定性（可以换成随机种子以增加多样性）
        setup_seed(2026) 
        
        # 自动测试模式下，打印当前正在测试的 Prompt
        if input_mode == 0: 
            print(f'💬: {prompt}')
            
        # 根据设置的携带历史轮数（args.historys）对对话历史进行切片截取
        # 比如 historys=2，则只保留最后 2 个元素（包含 1 轮 user 和 1 轮 assistant 问答）
        conversation = conversation[-args.historys:] if args.historys else []
        
        # 将当前用户的输入装入对话历史
        conversation.append({"role": "user", "content": prompt})

        # 构建应用聊天模版（Chat Template）的参数字典
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        
        # 如果当前使用的是推理/思考模型（'reason'），则开启思考模式支持（例如 DeepSeek 样式的思维链）
        if args.weight == 'reason': 
            templates["enable_thinking"] = True
            
        # 如果不是预训练（pretrain）阶段的权重（即已经过 SFT 对齐的模型），则套用 Chat Template
        # 如果是 pretrain 权重，直接在前面拼上 BOS 标志符（序列开始符）作为原始文本输入
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        
        # 将文本转换成模型能够识别的 PyTorch Tensor Tensor，并移动到 GPU/CPU
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()  # 记录生成开始的时间戳
        
        # 调用模型开始生成文本
        generated_ids = model.generate(
            inputs=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, # 最大新生成 Token 数量
            do_sample=True,                      # 启用采样模式（配合温度和 top_p 参数）
            streamer=streamer,                   # 使用流式传输器，边生成边打印到终端
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
            top_p=args.top_p, 
            temperature=args.temperature, 
            repetition_penalty=1.0               # 重复惩罚系数，1.0 表示不惩罚
        )
        
        # 从生成的完整 Token 序列中切片出“新生成的回复部分”，并解码为文本字符串
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        # 将模型的回复内容追加到历史对话中，用于下一轮多轮对话
        conversation.append({"role": "assistant", "content": response})
        
        # 计算本次模型实际生成的新 Token 数量
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        
        # 如果开启了速度显示，计算并打印每秒生成的 Token 速度
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n')
        else:
            print('\n\n')

if __name__ == "__main__":
    main()