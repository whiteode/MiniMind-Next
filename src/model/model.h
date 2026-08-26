#pragma once

#include <vector>
#include <string>
#include <cstdio>
#include <cstddef>
#include "model/embedding.h"

namespace minimind {

// 模型超参数
struct ModelConfig {
    int dim = 768;        // 隐藏层维度
    int hidden_dim = 2048;// FFN 维度
    int n_layers = 16;   // 层数
    int n_heads = 8;     // Q 头数
    int n_kv_heads = 2;  // KV 头数 (GQA)
    int vocab_size = 6400; // 词表大小
    int seq_len = 1024;  // 最大序列长度
};

// 单层 Transformer 权重指针
struct LayerWeights {
    float* rms_att_weight = nullptr; // [dim]
    float* wq = nullptr;             // [dim, dim]
    float* wk = nullptr;             // [kv_dim, dim]
    float* wv = nullptr;             // [kv_dim, dim]
    float* wo = nullptr;             // [dim, dim]
    float* rms_ffn_weight = nullptr; // [dim]
    float* w_gate = nullptr;         // [hidden_dim, dim]
    float* w_up = nullptr;           // [hidden_dim, dim]
    float* w_down = nullptr;         // [dim, hidden_dim]
};

// 模型运行态缓存（包含动态激活值与 KV Cache）
struct RunState {
    std::vector<float> x;      // 激活向量 [dim]
    std::vector<float> xb;     // 辅助缓存 [dim]
    std::vector<float> xb2;    // 辅助缓存2 [dim]
    std::vector<float> hb;     // FFN 激活 [hidden_dim]
    std::vector<float> hb2;    // FFN 激活2 [hidden_dim]
    std::vector<float> q;      // Query [dim]
    std::vector<float> k;      // Key [n_kv_heads * head_dim]
    std::vector<float> v;      // Value [n_kv_heads * head_dim]
    std::vector<float> att;    // Attention scores [n_heads, seq_len]
    std::vector<float> logits; // Logits 输出 [vocab_size]

    // KV Cache
    std::vector<float> key_cache;   // [n_layers, seq_len, kv_dim]
    std::vector<float> value_cache; // [n_layers, seq_len, kv_dim]

    explicit RunState(const ModelConfig& p);
    void reset();
};

class MiniMindModel {
public:
    ModelConfig config;

    // 权重指针 (Transformer 各层与 LM Head)
    std::vector<LayerWeights> layers;
    float* rms_final_weight = nullptr;
    float* wcls = nullptr;

    // 连续内存缓冲
    std::vector<float> weight_data;

    MiniMindModel();

    // 从二进制文件中读取权重矩阵并完成映射 (不含 Embedding)
    bool load_weights(FILE* f);

    // 单步 Transformer 前向推理计算，接收 embedding 模块进行首层 Token 查找
    void forward(RunState& state, const QuantizedEmbedding& embedding, int token, int pos) const;
};

} // namespace minimind
