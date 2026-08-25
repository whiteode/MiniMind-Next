import os
import argparse
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# 1. 量化方案类定义 (14 种量化算法)
# ==============================================================================

# 方案 1: NF4 动态 Codebook 量化
class QuantizedEmbeddingNF4(nn.Module):
    def __init__(self, original_embedding, bits=4):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.bits = bits
        self.n_bins = 2 ** bits

        weight = original_embedding.weight.data.float()
        self.codebook = self._generate_quantile_codebook(weight)

        diff = weight.unsqueeze(-1) - self.codebook.view(1, 1, -1)
        q_indices = torch.argmin(diff.abs(), dim=-1).to(torch.uint8)

        assert self.embedding_dim % 2 == 0
        q_reshaped = q_indices.view(self.num_embeddings, -1, 2)
        q_packed = (q_reshaped[:, :, 0] << 4) | q_reshaped[:, :, 1]

        self.register_buffer("q_weight", q_packed)
        self.register_buffer("dynamic_codebook", self.codebook.to(torch.float16))

    def _generate_quantile_codebook(self, weight):
        flat_weight = weight.view(-1)
        sorted_weight = torch.sort(flat_weight).values
        p = torch.linspace(0, 1, 2 * self.n_bins + 1, device=weight.device)[1::2]
        indices = (p * (len(sorted_weight) - 1)).long()
        return sorted_weight[indices]

    def forward(self, input_ids):
        packed = F.embedding(input_ids, self.q_weight)
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        q = torch.stack([high, low], dim=-1).view(*packed.shape[:-1], -1)
        return self.dynamic_codebook[q.long()]


# ----------------- INT4 方案 (对称量化, Range: -8 ~ 7) -----------------

class QuantizedEmbeddingInt4Tensor(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim

        weight = original_embedding.weight.data.float()
        abs_max = weight.abs().max()

        scale = abs_max.clamp(min=1e-6) / 7.0
        q_weight_int = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)

        q_weight_4b = (q_weight_int & 0xF).to(torch.uint8)
        assert self.embedding_dim % 2 == 0
        q_reshaped = q_weight_4b.view(self.num_embeddings, -1, 2)
        q_packed = (q_reshaped[:, :, 0] << 4) | q_reshaped[:, :, 1]

        self.register_buffer("q_weight", q_packed)
        self.register_buffer("scale", scale.to(torch.float16))

    def forward(self, input_ids):
        packed = F.embedding(input_ids, self.q_weight)
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        q = torch.stack([high, low], dim=-1).view(*packed.shape[:-1], -1).to(torch.int8)
        q[q >= 8] -= 16
        return q.to(torch.float16) * self.scale


class QuantizedEmbeddingInt4Token(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        assert self.embedding_dim % 2 == 0

        weight = original_embedding.weight.data.float()
        abs_max = weight.abs().max(dim=-1, keepdim=True)[0]

        scales = abs_max.clamp(min=1e-6) / 7.0
        q_weight_int = torch.round(weight / scales).clamp(-8, 7).to(torch.int8)

        q_weight_4b = (q_weight_int & 0xF).to(torch.uint8)
        q_reshaped = q_weight_4b.view(self.num_embeddings, -1, 2)
        q_packed = (q_reshaped[:, :, 0] << 4) | q_reshaped[:, :, 1]

        self.register_buffer("q_weight", q_packed)
        self.register_buffer("scales", scales.to(torch.float16))

    def forward(self, input_ids):
        packed = F.embedding(input_ids, self.q_weight)
        s_val = F.embedding(input_ids, self.scales)

        high = (packed >> 4) & 0xF
        low = packed & 0xF
        q = torch.stack([high, low], dim=-1).view(*packed.shape[:-1], -1).to(torch.int8)
        q[q >= 8] -= 16

        return q.to(torch.float16) * s_val


class QuantizedEmbeddingInt4Group(nn.Module):
    def __init__(self, original_embedding, group_size=64):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.group_size = group_size

        assert self.embedding_dim % group_size == 0
        assert group_size % 2 == 0

        weight = original_embedding.weight.data.float()
        V, D = weight.shape
        G = D // group_size

        weight_grouped = weight.view(V, G, group_size)
        abs_max = weight_grouped.abs().max(dim=-1, keepdim=True)[0]

        scales = abs_max.clamp(min=1e-6) / 7.0
        q_weight_int = torch.round(weight_grouped / scales).clamp(-8, 7).to(torch.int8)

        q_weight_4b = (q_weight_int & 0xF).to(torch.uint8)
        q_reshaped = q_weight_4b.view(V, D // 2, 2)
        q_packed = (q_reshaped[:, :, 0] << 4) | q_reshaped[:, :, 1]

        self.register_buffer("q_weight", q_packed)
        self.register_buffer("scales", scales.to(torch.float16).view(V, G))

    def forward(self, input_ids):
        G = self.embedding_dim // self.group_size
        packed = F.embedding(input_ids, self.q_weight)

        high = (packed >> 4) & 0xF
        low = packed & 0xF
        q = torch.stack([high, low], dim=-1).view(*packed.shape[:-1], G, self.group_size).to(torch.int8)
        q[q >= 8] -= 16

        q_val = q.to(torch.float16)
        s_val = F.embedding(input_ids, self.scales).unsqueeze(-1)

        out = q_val * s_val
        return out.view(*packed.shape[:-1], self.embedding_dim)


# ----------------- UINT4 方案 (非对称量化, Range: 0 ~ 15) -----------------

class QuantizedEmbeddingUint4Tensor(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim

        weight = original_embedding.weight.data.float()
        min_val, max_val = weight.min(), weight.max()

        scale = (max_val - min_val).clamp(min=1e-6) / 15.0
        zero_point = torch.round(-min_val / scale).clamp(0, 15)

        q_weight = torch.round(weight / scale + zero_point).clamp(0, 15).to(torch.uint8)
        assert self.embedding_dim % 2 == 0
        q_weight = q_weight.view(self.num_embeddings, -1, 2)
        q_packed = (q_weight[:, :, 0] << 4) | q_weight[:, :, 1]

        self.register_buffer("q_weight", q_packed)
        self.register_buffer("scale", scale.to(torch.float16))
        self.register_buffer("zero_point", zero_point.to(torch.float16))

    def forward(self, input_ids):
        packed = F.embedding(input_ids, self.q_weight)
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        q = torch.stack([high, low], dim=-1).view(*packed.shape[:-1], -1).to(torch.float16)
        return (q - self.zero_point) * self.scale


class QuantizedEmbeddingUint4Token(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        assert self.embedding_dim % 2 == 0

        weight = original_embedding.weight.data.float()
        min_val = weight.min(dim=-1, keepdim=True)[0]
        max_val = weight.max(dim=-1, keepdim=True)[0]

        scales = (max_val - min_val).clamp(min=1e-6) / 15.0
        zero_points = torch.round(-min_val / scales).clamp(0, 15)

        q_weight = torch.round(weight / scales + zero_points).clamp(0, 15).to(torch.uint8)
        q_reshaped = q_weight.view(self.num_embeddings, -1, 2)
        q_packed = (q_reshaped[:, :, 0] << 4) | q_reshaped[:, :, 1]

        self.register_buffer("q_weight", q_packed)
        self.register_buffer("scales", scales.to(torch.float16))
        self.register_buffer("zero_points", zero_points.to(torch.float16))

    def forward(self, input_ids):
        packed = F.embedding(input_ids, self.q_weight)
        s_val = F.embedding(input_ids, self.scales)
        z_val = F.embedding(input_ids, self.zero_points)

        high = (packed >> 4) & 0xF
        low = packed & 0xF
        q = torch.stack([high, low], dim=-1).view(*packed.shape[:-1], -1).to(torch.float16)
        return (q - z_val) * s_val


class QuantizedEmbeddingUint4Group(nn.Module):
    def __init__(self, original_embedding, group_size=64):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.group_size = group_size

        assert self.embedding_dim % group_size == 0
        assert group_size % 2 == 0

        weight = original_embedding.weight.data.float()
        V, D = weight.shape
        G = D // group_size

        weight_grouped = weight.view(V, G, group_size)
        min_val = weight_grouped.min(dim=-1, keepdim=True)[0]
        max_val = weight_grouped.max(dim=-1, keepdim=True)[0]

        scales = (max_val - min_val).clamp(min=1e-6) / 15.0
        zero_points = torch.round(-min_val / scales).clamp(0, 15)

        q_weight = torch.round(weight_grouped / scales + zero_points).clamp(0, 15).to(torch.uint8)
        q_reshaped = q_weight.view(V, D // 2, 2)
        q_packed = (q_reshaped[:, :, 0] << 4) | q_reshaped[:, :, 1]

        self.register_buffer("q_weight", q_packed)
        self.register_buffer("scales", scales.to(torch.float16).view(V, G))
        self.register_buffer("zero_points", zero_points.to(torch.float16).view(V, G))

    def forward(self, input_ids):
        G = self.embedding_dim // self.group_size
        packed = F.embedding(input_ids, self.q_weight)

        high = (packed >> 4) & 0xF
        low = packed & 0xF
        q_val = torch.stack([high, low], dim=-1).view(*packed.shape[:-1], G, self.group_size).to(torch.float16)

        s_val = F.embedding(input_ids, self.scales).unsqueeze(-1)
        z_val = F.embedding(input_ids, self.zero_points).unsqueeze(-1)

        out = (q_val - z_val) * s_val
        return out.view(*packed.shape[:-1], self.embedding_dim)


# ----------------- INT8 方案 (对称量化, Range: -128 ~ 127) -----------------

class QuantizedEmbeddingInt8Tensor(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        weight = original_embedding.weight.data.float()

        abs_max = weight.abs().max()
        scale = abs_max.clamp(min=1e-6) / 127.0

        q_weight = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale.to(torch.float16))

    def forward(self, input_ids):
        q_val = F.embedding(input_ids, self.q_weight).to(torch.float16)
        return q_val * self.scale


class QuantizedEmbeddingInt8Token(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        weight = original_embedding.weight.data.float()

        abs_max = weight.abs().max(dim=-1, keepdim=True)[0]
        scales = abs_max.clamp(min=1e-6) / 127.0

        q_weight = torch.round(weight / scales).clamp(-128, 127).to(torch.int8)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scales", scales.to(torch.float16))

    def forward(self, input_ids):
        q_val = F.embedding(input_ids, self.q_weight).to(torch.float16)
        s_val = F.embedding(input_ids, self.scales)
        return q_val * s_val


class QuantizedEmbeddingInt8Group(nn.Module):
    def __init__(self, original_embedding, group_size=64):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.group_size = group_size

        weight = original_embedding.weight.data.float()
        V, D = weight.shape
        assert D % group_size == 0
        G = D // group_size

        weight_grouped = weight.view(V, G, group_size)
        abs_max = weight_grouped.abs().max(dim=-1, keepdim=True)[0]

        scales = abs_max.clamp(min=1e-6) / 127.0
        q_weight = torch.round(weight_grouped / scales).clamp(-128, 127).to(torch.int8)

        self.register_buffer("q_weight", q_weight.view(V, D))
        self.register_buffer("scales", scales.to(torch.float16))

    def forward(self, input_ids):
        V, D = self.q_weight.shape
        G = self.embedding_dim // self.group_size

        q_val = F.embedding(input_ids, self.q_weight).to(torch.float16)
        q_val = q_val.view(*q_val.shape[:-1], G, self.group_size)

        scales_2d = self.scales.view(V, G)
        s_val = F.embedding(input_ids, scales_2d).unsqueeze(-1)

        out = q_val * s_val
        return out.view(*out.shape[:-2], D)


# ----------------- UINT8 方案 (非对称量化, Range: 0 ~ 255) -----------------

class QuantizedEmbeddingUint8Tensor(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        weight = original_embedding.weight.data.float()

        min_val, max_val = weight.min(), weight.max()
        scale = (max_val - min_val).clamp(min=1e-6) / 255.0
        zero_point = torch.round(-min_val / scale).clamp(0, 255)

        q_weight = torch.round(weight / scale + zero_point).clamp(0, 255).to(torch.uint8)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale.to(torch.float16))
        self.register_buffer("zero_point", zero_point.to(torch.float16))

    def forward(self, input_ids):
        q_val = F.embedding(input_ids, self.q_weight).to(torch.float16)
        return (q_val - self.zero_point) * self.scale


class QuantizedEmbeddingUint8Token(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        weight = original_embedding.weight.data.float()

        min_val = weight.min(dim=-1, keepdim=True)[0]
        max_val = weight.max(dim=-1, keepdim=True)[0]

        scales = (max_val - min_val).clamp(min=1e-6) / 255.0
        zero_points = torch.round(-min_val / scales).clamp(0, 255)

        q_weight = torch.round(weight / scales + zero_points).clamp(0, 255).to(torch.uint8)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scales", scales.to(torch.float16))
        self.register_buffer("zero_points", zero_points.to(torch.float16))

    def forward(self, input_ids):
        q_val = F.embedding(input_ids, self.q_weight).to(torch.float16)
        s_val = F.embedding(input_ids, self.scales)
        z_val = F.embedding(input_ids, self.zero_points)
        return (q_val - z_val) * s_val


class QuantizedEmbeddingUint8Group(nn.Module):
    def __init__(self, original_embedding, group_size=64):
        super().__init__()
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.group_size = group_size

        weight = original_embedding.weight.data.float()
        V, D = weight.shape
        assert D % group_size == 0
        G = D // group_size

        weight_grouped = weight.view(V, G, group_size)
        min_val = weight_grouped.min(dim=-1, keepdim=True)[0]
        max_val = weight_grouped.max(dim=-1, keepdim=True)[0]

        scales = (max_val - min_val).clamp(min=1e-6) / 255.0
        zero_points = torch.round(-min_val / scales).clamp(0, 255)

        q_weight = torch.round(weight_grouped / scales + zero_points).clamp(0, 255).to(torch.uint8)
        self.register_buffer("q_weight", q_weight.view(V, D))
        self.register_buffer("scales", scales.to(torch.float16).view(V, G))
        self.register_buffer("zero_points", zero_points.to(torch.float16).view(V, G))

    def forward(self, input_ids):
        V, D = self.q_weight.shape
        G = self.embedding_dim // self.group_size

        q_val = F.embedding(input_ids, self.q_weight).to(torch.float16)
        q_val = q_val.view(*q_val.shape[:-1], G, self.group_size)

        scales_2d = self.scales.view(V, G)
        zeros_2d = self.zero_points.view(V, G)

        s_val = F.embedding(input_ids, scales_2d).unsqueeze(-1)
        z_val = F.embedding(input_ids, zeros_2d).unsqueeze(-1)

        out = (q_val - z_val) * s_val
        return out.view(*out.shape[:-2], D)


# ==============================================================================
# 2. 导出与序列化函数
# ==============================================================================

QUANT_TYPE_MAP = {
    "fp16": 0,
    "nf4_tensor": 1,
    "int4_tensor": 2, "int4_token": 3, "int4_group": 4,
    "uint4_tensor": 5, "uint4_token": 6, "uint4_group": 7,
    "int8_tensor": 8, "int8_token": 9, "int8_group": 10,
    "uint8_tensor": 11, "uint8_token": 12, "uint8_group": 13
}

QUANT_CLASS_MAP = {
    "nf4_tensor": QuantizedEmbeddingNF4,
    "int4_tensor": QuantizedEmbeddingInt4Tensor,
    "int4_token": QuantizedEmbeddingInt4Token,
    "int4_group": QuantizedEmbeddingInt4Group,
    "uint4_tensor": QuantizedEmbeddingUint4Tensor,
    "uint4_token": QuantizedEmbeddingUint4Token,
    "uint4_group": QuantizedEmbeddingUint4Group,
    "int8_tensor": QuantizedEmbeddingInt8Tensor,
    "int8_token": QuantizedEmbeddingInt8Token,
    "int8_group": QuantizedEmbeddingInt8Group,
    "uint8_tensor": QuantizedEmbeddingUint8Tensor,
    "uint8_token": QuantizedEmbeddingUint8Token,
    "uint8_group": QuantizedEmbeddingUint8Group
}

def export_to_quantized_embedding(quantized_model, method_name, save_path, group_size=0):
    """
    将量化后(或原始 fp16)的 Embedding 导出为 .embedding 二进制文件。
    所有 float 参数均存为 float16 格式。
    """
    if method_name not in QUANT_TYPE_MAP:
        raise ValueError(f"暂不支持导出该量化方法: {method_name}")

    quant_type_id = QUANT_TYPE_MAP[method_name]

    if method_name == "fp16":
        num_embeddings = quantized_model.weight.shape[0]
        embedding_dim = quantized_model.weight.shape[1]
    else:
        num_embeddings = quantized_model.num_embeddings
        embedding_dim = quantized_model.embedding_dim

    print(f"[{method_name}] 开始导出到 {save_path} ...")

    with open(save_path, "wb") as f:
        # 1. Magic Word (MMEM: MiniMind Embedding)
        f.write(b"MMEM")
        # 2. Version (uint32 = 1)
        f.write(struct.pack("<I", 1))
        # 3. QuantType ID (uint32)
        f.write(struct.pack("<I", quant_type_id))
        # 4. Vocab Size & Dim (uint32, uint32)
        f.write(struct.pack("<I", num_embeddings))
        f.write(struct.pack("<I", embedding_dim))
        # 5. Group Size (uint32)
        f.write(struct.pack("<I", group_size))

        def write_tensor(tensor, np_dtype):
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy().astype(np_dtype)
            else:
                arr = np.array(tensor, dtype=np_dtype)
            f.write(arr.tobytes())

        # Payload
        if method_name == "fp16":
            write_tensor(quantized_model.weight, np.float16)

        elif method_name == "nf4_tensor":
            write_tensor(quantized_model.q_weight, np.uint8)
            write_tensor(quantized_model.dynamic_codebook, np.float16)

        elif method_name in ["int4_tensor", "int4_token", "int4_group"]:
            write_tensor(quantized_model.q_weight, np.uint8)
            scales = quantized_model.scale if hasattr(quantized_model, "scale") else quantized_model.scales
            write_tensor(scales, np.float16)

        elif method_name in ["uint4_tensor", "uint4_token", "uint4_group"]:
            write_tensor(quantized_model.q_weight, np.uint8)
            scales = quantized_model.scale if hasattr(quantized_model, "scale") else quantized_model.scales
            zeros = quantized_model.zero_point if hasattr(quantized_model, "zero_point") else quantized_model.zero_points
            write_tensor(scales, np.float16)
            write_tensor(zeros, np.float16)

        elif method_name in ["int8_tensor", "int8_token", "int8_group"]:
            write_tensor(quantized_model.q_weight, np.int8)
            scales = quantized_model.scale if hasattr(quantized_model, "scale") else quantized_model.scales
            write_tensor(scales, np.float16)

        elif method_name in ["uint8_tensor", "uint8_token", "uint8_group"]:
            write_tensor(quantized_model.q_weight, np.uint8)
            scales = quantized_model.scale if hasattr(quantized_model, "scale") else quantized_model.scales
            zeros = quantized_model.zero_point if hasattr(quantized_model, "zero_point") else quantized_model.zero_points
            write_tensor(scales, np.float16)
            write_tensor(zeros, np.float16)

    print(f"[{method_name}] 导出完成! 大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")


def load_embedding_layer(model_path):
    """
    自动从 HuggingFace 目录或 PyTorch .pth 权重中加载 embedding 层
    """
    if os.path.isdir(model_path):
        print(f">>> 正在从 HuggingFace 目录加载模型: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte
        elif hasattr(model, "tok_embeddings"):
            return model.tok_embeddings
        else:
            # 搜索第一个 nn.Embedding 模块
            for mod in model.modules():
                if isinstance(mod, nn.Embedding):
                    return mod
            raise RuntimeError(f"在 {model_path} 中未找到 embedding 层")
    elif os.path.isfile(model_path) and model_path.endswith(".pth"):
        print(f">>> 正在从 PyTorch 检查点加载权重: {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # 寻找 tok_embeddings 或 embed_tokens 权重
        for k, v in state_dict.items():
            if "tok_embeddings.weight" in k or "embed_tokens.weight" in k:
                vocab_size, dim = v.shape
                emb = nn.Embedding(vocab_size, dim)
                emb.weight.data.copy_(v.float())
                return emb
        raise RuntimeError(f"在 {model_path} 中未找到 embedding 权重")
    else:
        raise ValueError(f"无法识别的模型路径: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Embedding layer into quantized .embedding binary format")
    parser.add_argument("--model_path", type=str, default="resource/MiniMind2", help="Path to HuggingFace directory or PyTorch .pth")
    parser.add_argument("--output_dir", type=str, default="models/embedding", help="Directory to save .embedding files")
    parser.add_argument("--methods", type=str, nargs="+", default=["fp16", "int8_token", "int4_group"],
                        help="Quantization methods to export (or 'all')")
    parser.add_argument("--group_size", type=int, default=64, help="Group size for group quantization")
    args = parser.parse_args()

    emb = load_embedding_layer(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    methods = args.methods
    if "all" in methods:
        methods = ["fp16"] + list(QUANT_CLASS_MAP.keys())

    print(f"\nEmbedding 形状: Vocab Size = {emb.num_embeddings}, Dim = {emb.embedding_dim}")
    print(f"导出格式列表: {methods}\n")

    for method in methods:
        save_path = os.path.join(args.output_dir, f"embedding_{method}.embedding")
        if method == "fp16":
            export_to_quantized_embedding(emb, "fp16", save_path, group_size=0)
        else:
            QuantClass = QUANT_CLASS_MAP[method]
            is_group = "group" in method
            gs = args.group_size if is_group else 0
            if is_group:
                quant_model = QuantClass(emb, group_size=gs)
            else:
                quant_model = QuantClass(emb)
            export_to_quantized_embedding(quant_model, method, save_path, group_size=gs)

    print("\n✅ Embedding 量化导出全部完成！")


if __name__ == "__main__":
    main()
