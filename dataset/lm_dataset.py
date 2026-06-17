from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    """
    大语言模型（LLM）自监督预训练（Pre-training）专用的数据集处理类。
    继承自 PyTorch 的 Dataset 基类，负责将原始文本转换为模型可直接训练的 Tensor 序列。
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化函数：加载数据源并配置分词参数。
        
        参数:
        - data_path: 原始 JSON 格式训练数据文件的路径
        - tokenizer: 绑定的分词器（Tokenizer）实例
        - max_length: 文本最大截断长度（上下文窗口大小，默认512）
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 使用 Hugging Face datasets 库流式/直接加载本地的 JSON 格式数据集
        # 读取的数据通常是一个字典结构，包含类似 {"text": "今天天气真好..."} 的字段
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        """
        魔术方法：返回整个数据集中样本的总数量。
        PyTorch 的 DataLoader 在划分 Batch 和计算 Epoch 时会调用它。
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        魔术方法：根据索引读取单条样本，并进行 Tokenize、拼接特殊符号、Padding、生成标签等全套核心处理。
        
        参数:
        - index: 当前需要读取的样本索引
        返回:
        - input_ids: 模型输入的 Token ID 序列（Tensor）
        - labels: 用于计算交叉熵损失（Loss）的标签序列（Tensor）
        """
        # 1. 根据索引从数据集中提取单条样本字典
        sample = self.samples[index]
        
        # 2. 将样本中的 'text' 字段转化为字符串，并利用分词器转换为一维 Token ID 序列
        # add_special_tokens=False: 暂时先不让分词器自动加 BOS/EOS，后面我们自己手动加
        # max_length=self.max_length - 2: 预留2个位置给头部的 BOS 和尾部的 EOS，防止加上后超长
        # truncation=True: 超过预留长度的文本直接无情截断
        tokens = self.tokenizer(
            str(sample['text']), 
            add_special_tokens=False, 
            max_length=self.max_length - 2, 
            truncation=True
        ).input_ids
        
        # 3. 手动包裹特殊符号
        # 在序列开头拼上 BOS（文本开始符，通常是 <s>），在末尾拼上 EOS（文本结束符，通常是 </s>）
        # 这是为了让大模型学会如何识别一篇文章的开头和结尾
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # 4. 填充 Padding 操作（对齐长度）
        # 如果当前文本长度小于给定的最大长度 max_length，用 pad_token_id（填充符）在右侧补齐
        # 计算公式：当前 tokens 后面，拼接 (max_length - 当前长度) 个填充数字
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        # 5. 将 Python 的普通的 List 列表转换为 PyTorch 的张量（Tensor）
        # dtype=torch.long: 必须是 64 位长整型，因为这是 Embedding 嵌入层要求的索引格式
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 6. 生成自监督训练的标签（Labels）
        # 自回归语言模型的任务是“预测下一个词”，它的标签最初和输入序列是一模一样的（克隆一份）
        labels = input_ids.clone()
        
        # 7. 屏蔽遮掩 Padding 的损失计算
        # input_ids == self.tokenizer.pad_token_id 会生成一个布尔矩阵，定位到所有是 Padding 的位置
        # 将这些位置的标签强行修改为 -100。
        # 核心原因：PyTorch的交叉熵损失函数（nn.CrossEntropyLoss）默认会忽略掉标签值为 -100 的位置。
        # 这样做能保证模型只对“真实文本”计算 Loss，而不会去痛苦地学习和预测那些用来凑长度的无意义填充符
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        # 返回最终成对的“输入”与“标签”，喂给模型训练
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }

if __name__ == "__main__":
    pass