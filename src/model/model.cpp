#include "model.h"
#include "../ops/math_ops.h"
#include <cstring>
#include <iostream>

namespace minimind {

RunState::RunState(const ModelConfig& p) {
    int head_dim = p.dim / p.n_heads;
    int kv_dim = p.n_kv_heads * head_dim;
    x.resize(p.dim);
    xb.resize(p.dim);
    xb2.resize(p.dim);
    hb.resize(p.hidden_dim);
    hb2.resize(p.hidden_dim);
    q.resize(p.dim);
    k.resize(kv_dim);
    v.resize(kv_dim);
    att.resize(p.n_heads * p.seq_len);
    logits.resize(p.vocab_size);
    key_cache.resize((size_t)p.n_layers * p.seq_len * kv_dim);
    value_cache.resize((size_t)p.n_layers * p.seq_len * kv_dim);
}

void RunState::reset() {
    std::fill(key_cache.begin(), key_cache.end(), 0.0f);
    std::fill(value_cache.begin(), value_cache.end(), 0.0f);
}

MiniMindModel::MiniMindModel() = default;

bool MiniMindModel::load_weights(FILE* f) {
    long weights_start = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, weights_start, SEEK_SET);

    size_t weights_bytes = file_size - weights_start;
    weight_data.resize(weights_bytes / sizeof(float));
    if (fread(weight_data.data(), sizeof(float), weight_data.size(), f) != weight_data.size()) {
        return false;
    }

    float* ptr = weight_data.data();
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int n_layers = config.n_layers;
    int kv_dim = config.n_kv_heads * (dim / config.n_heads);

    token_embedding_table = ptr; ptr += (size_t)config.vocab_size * dim;
    layers.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        layers[i].rms_att_weight = ptr; ptr += dim;
        layers[i].wq = ptr; ptr += dim * dim;
        layers[i].wk = ptr; ptr += kv_dim * dim;
        layers[i].wv = ptr; ptr += kv_dim * dim;
        layers[i].wo = ptr; ptr += dim * dim;
        layers[i].rms_ffn_weight = ptr; ptr += dim;
        layers[i].w_gate = ptr; ptr += hidden_dim * dim;
        layers[i].w_up = ptr; ptr += hidden_dim * dim;
        layers[i].w_down = ptr; ptr += dim * hidden_dim;
    }
    rms_final_weight = ptr; ptr += dim;
    wcls = ptr; ptr += (size_t)config.vocab_size * dim;

    return true;
}

void MiniMindModel::forward(RunState& s, int token, int pos) const {
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_dim = dim / config.n_heads;
    int kv_dim = config.n_kv_heads * head_dim;
    int kv_mul = config.n_heads / config.n_kv_heads; // GQA 倍率 (8 / 2 = 4)

    // 1. 获取 Embedding 向量: s.x = token_embedding_table[token]
    const float* token_emb = token_embedding_table + token * dim;
    std::memcpy(s.x.data(), token_emb, dim * sizeof(float));

    // 2. 遍历各层 Decoder Block
    for (int l = 0; l < config.n_layers; ++l) {
        const LayerWeights& lw = layers[l];

        // 2.1 Attention RMSNorm
        ops::rmsnorm(s.xb.data(), s.x.data(), lw.rms_att_weight, dim);

        // 2.2 Q, K, V 线性投影
        ops::matmul(s.q.data(), s.xb.data(), lw.wq, dim, dim);
        ops::matmul(s.k.data(), s.xb.data(), lw.wk, dim, kv_dim);
        ops::matmul(s.v.data(), s.xb.data(), lw.wv, dim, kv_dim);

        // 2.3 RoPE 旋转位置编码 (rotate_half)
        for (int h = 0; h < config.n_heads; ++h) {
            ops::apply_rope(s.q.data() + h * head_dim, head_dim, pos);
        }
        for (int h = 0; h < config.n_kv_heads; ++h) {
            ops::apply_rope(s.k.data() + h * head_dim, head_dim, pos);
        }

        // 2.4 保存至当前层 KV Cache
        size_t cache_offset = (size_t)l * config.seq_len * kv_dim + pos * kv_dim;
        std::memcpy(s.key_cache.data() + cache_offset, s.k.data(), kv_dim * sizeof(float));
        std::memcpy(s.value_cache.data() + cache_offset, s.v.data(), kv_dim * sizeof(float));

        // 2.5 Grouped-Query Attention
        for (int h = 0; h < config.n_heads; ++h) {
            float* q_head = s.q.data() + h * head_dim;
            float* att_head = s.att.data() + h * config.seq_len;
            int kv_h = h / kv_mul;

            // 计算注意力分数 Q * K^T / sqrt(head_dim)
            for (int t = 0; t <= pos; ++t) {
                const float* k_head = s.key_cache.data() + (size_t)l * config.seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q_head[i] * k_head[i];
                }
                score /= std::sqrt((float)head_dim);
                att_head[t] = score;
            }

            // Softmax
            ops::softmax(att_head, pos + 1);

            // 加权求和 Attention * V 存入 s.xb
            float* xb_head = s.xb.data() + h * head_dim;
            std::memset(xb_head, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; ++t) {
                const float* v_head = s.value_cache.data() + (size_t)l * config.seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                float a = att_head[t];
                for (int i = 0; i < head_dim; ++i) {
                    xb_head[i] += a * v_head[i];
                }
            }
        }

        // 2.6 Out Projection
        ops::matmul(s.xb2.data(), s.xb.data(), lw.wo, dim, dim);

        // 2.7 残差连接 1
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];

        // 2.8 FFN RMSNorm
        ops::rmsnorm(s.xb.data(), s.x.data(), lw.rms_ffn_weight, dim);

        // 2.9 SwiGLU: silu(w_gate * xb) * (w_up * xb)
        ops::matmul(s.hb.data(), s.xb.data(), lw.w_gate, dim, hidden_dim);
        ops::matmul(s.hb2.data(), s.xb.data(), lw.w_up, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; ++i) {
            float val = s.hb[i];
            float silu = val / (1.0f + std::exp(-val));
            s.hb[i] = silu * s.hb2[i];
        }

        // 2.10 Down Projection
        ops::matmul(s.xb.data(), s.hb.data(), lw.w_down, hidden_dim, dim);

        // 2.11 残差连接 2
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];
    }

    // 3. Final RMSNorm
    ops::rmsnorm(s.x.data(), s.x.data(), rms_final_weight, dim);

    // 4. LM Head 分类输出 Logits
    ops::matmul(s.logits.data(), s.x.data(), wcls, dim, config.vocab_size);
}

} // namespace minimind
