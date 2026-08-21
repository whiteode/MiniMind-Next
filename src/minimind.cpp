#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// ----------------------------------------------------------------------------
// 1. 配置参数与权重结构体
// ----------------------------------------------------------------------------

struct Config {
    int dim;        // 隐藏层维度 (768)
    int hidden_dim; // FFN 维度 (2048)
    int n_layers;   // 层数 (16)
    int n_heads;    // Q 头数 (8)
    int n_kv_heads; // KV 头数 (2)
    int vocab_size; // 词表大小 (6400)
    int seq_len;    // 最大序列长度 (1024)
};

struct LayerWeights {
    float* rms_att_weight; // [dim]
    float* wq;             // [dim, dim]
    float* wk;             // [kv_dim, dim]
    float* wv;             // [kv_dim, dim]
    float* wo;             // [dim, dim]
    float* rms_ffn_weight; // [dim]
    float* w_gate;         // [hidden_dim, dim]
    float* w_up;           // [hidden_dim, dim]
    float* w_down;         // [dim, hidden_dim]
};

struct TransformerWeights {
    float* token_embedding_table; // [vocab_size, dim]
    std::vector<LayerWeights> layers;
    float* rms_final_weight;      // [dim]
    float* wcls;                  // [vocab_size, dim]
};

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
    std::vector<float> key_cache;   // [n_layers, seq_len, n_kv_heads * head_dim]
    std::vector<float> value_cache; // [n_layers, seq_len, n_kv_heads * head_dim]

    RunState(const Config& p) {
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
};

// ----------------------------------------------------------------------------
// 2. Tokenizer (BPE / 字典树贪心匹配)
// ----------------------------------------------------------------------------

struct Tokenizer {
    std::vector<std::string> raw_vocab;
    std::vector<std::string> decode_vocab;
    int vocab_size;

    Tokenizer(int v_size) : vocab_size(v_size), raw_vocab(v_size), decode_vocab(v_size) {}

    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        size_t i = 0;
        while (i < text.size()) {
            int best_id = -1;
            size_t best_len = 0;
            // 匹配 decode_vocab 中最长的 token
            for (int id = 0; id < vocab_size; ++id) {
                const std::string& token = decode_vocab[id];
                if (token.empty()) continue;
                if (text.compare(i, token.size(), token) == 0) {
                    if (token.size() > best_len) {
                        best_len = token.size();
                        best_id = id;
                    }
                }
            }
            if (best_id != -1) {
                tokens.push_back(best_id);
                i += best_len;
            } else {
                i += 1;
            }
        }
        return tokens;
    }

    std::string decode(int token_id) {
        if (token_id >= 0 && token_id < vocab_size) {
            return decode_vocab[token_id];
        }
        return "";
    }
};

// ----------------------------------------------------------------------------
// 3. 核心计算算子 (RMSNorm, RoPE, Matmul, Softmax, SwiGLU)
// ----------------------------------------------------------------------------

void rmsnorm(float* o, const float* x, const float* weight, int size, float eps = 1e-5f) {
    float ss = 0.0f;
    for (int j = 0; j < size; ++j) ss += x[j] * x[j];
    ss /= size;
    ss += eps;
    float rsqrt_ss = 1.0f / std::sqrt(ss);
    for (int j = 0; j < size; ++j) {
        o[j] = weight[j] * (rsqrt_ss * x[j]);
    }
}

void matmul(float* xout, const float* x, const float* w, int n, int d) {
    // xout: [d], x: [n], w: [d, n] (行优先存储)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        float val = 0.0f;
        const float* w_row = w + i * n;
        for (int j = 0; j < n; ++j) {
            val += w_row[j] * x[j];
        }
        xout[i] = val;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

void apply_rope(float* vec, int head_dim, int pos, float rope_theta = 1000000.0f) {
    int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / std::pow(rope_theta, (float)(i * 2) / (float)head_dim);
        float val = pos * freq;
        float fcr = std::cos(val);
        float fci = std::sin(val);
        float v0 = vec[i];
        float v1 = vec[i + half_dim];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + half_dim] = v0 * fci + v1 * fcr;
    }
}

// ----------------------------------------------------------------------------
// 4. Transformer 前向推理
// ----------------------------------------------------------------------------

void forward(const Config& p, const TransformerWeights& w, RunState& s, int token, int pos) {
    int dim = p.dim;
    int hidden_dim = p.hidden_dim;
    int head_dim = dim / p.n_heads;
    int kv_dim = p.n_kv_heads * head_dim;
    int kv_mul = p.n_heads / p.n_kv_heads; // GQA 分组倍率 (8 / 2 = 4)

    // 1. 获取当前 Token 的 Embedding: s.x = w.token_embedding_table[token]
    const float* token_emb = w.token_embedding_table + token * dim;
    std::memcpy(s.x.data(), token_emb, dim * sizeof(float));

    // 2. 遍历所有 Transformer 层
    for (int l = 0; l < p.n_layers; ++l) {
        const LayerWeights& lw = w.layers[l];

        // Attention RMSNorm
        rmsnorm(s.xb.data(), s.x.data(), lw.rms_att_weight, dim);

        // Q, K, V 投影
        matmul(s.q.data(), s.xb.data(), lw.wq, dim, dim);
        matmul(s.k.data(), s.xb.data(), lw.wk, dim, kv_dim);
        matmul(s.v.data(), s.xb.data(), lw.wv, dim, kv_dim);

        // 应用 RoPE 旋转位置编码 (rotate_half)
        for (int h = 0; h < p.n_heads; ++h) {
            apply_rope(s.q.data() + h * head_dim, head_dim, pos);
        }
        for (int h = 0; h < p.n_kv_heads; ++h) {
            apply_rope(s.k.data() + h * head_dim, head_dim, pos);
        }

        // 保存 K, V 到 KV Cache
        size_t cache_offset = (size_t)l * p.seq_len * kv_dim + pos * kv_dim;
        std::memcpy(s.key_cache.data() + cache_offset, s.k.data(), kv_dim * sizeof(float));
        std::memcpy(s.value_cache.data() + cache_offset, s.v.data(), kv_dim * sizeof(float));

        // Grouped-Query Attention 计算
        for (int h = 0; h < p.n_heads; ++h) {
            float* q_head = s.q.data() + h * head_dim;
            float* att_head = s.att.data() + h * p.seq_len;
            int kv_h = h / kv_mul; // 映射到对应的 KV 头

            // 计算与历史各 step 的 Q * K^T 得分
            for (int t = 0; t <= pos; ++t) {
                const float* k_head = s.key_cache.data() + (size_t)l * p.seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q_head[i] * k_head[i];
                }
                score /= std::sqrt((float)head_dim);
                att_head[t] = score;
            }

            // 对有效时间步做 Softmax
            softmax(att_head, pos + 1);

            // 加权求和 Attention * V 写入 s.xb
            float* xb_head = s.xb.data() + h * head_dim;
            std::memset(xb_head, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; ++t) {
                const float* v_head = s.value_cache.data() + (size_t)l * p.seq_len * kv_dim + t * kv_dim + kv_h * head_dim;
                float a = att_head[t];
                for (int i = 0; i < head_dim; ++i) {
                    xb_head[i] += a * v_head[i];
                }
            }
        }

        // Output Projection: xb2 = wo * xb
        matmul(s.xb2.data(), s.xb.data(), lw.wo, dim, dim);

        // 残差连接 1: x = x + xb2
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];

        // FFN RMSNorm: xb = norm(x)
        rmsnorm(s.xb.data(), s.x.data(), lw.rms_ffn_weight, dim);

        // SwiGLU: hb = silu(w_gate * xb) * (w_up * xb)
        matmul(s.hb.data(), s.xb.data(), lw.w_gate, dim, hidden_dim);
        matmul(s.hb2.data(), s.xb.data(), lw.w_up, dim, hidden_dim);

        // SiLU 激活与逐元素点乘
        for (int i = 0; i < hidden_dim; ++i) {
            float val = s.hb[i];
            float silu = val / (1.0f + std::exp(-val));
            s.hb[i] = silu * s.hb2[i];
        }

        // Down Projection: xb = w_down * hb
        matmul(s.xb.data(), s.hb.data(), lw.w_down, hidden_dim, dim);

        // 残差连接 2: x = x + xb
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];
    }

    // 3. 最终 RMSNorm
    rmsnorm(s.x.data(), s.x.data(), w.rms_final_weight, dim);

    // 4. LM Head 分类输出 Logits
    matmul(s.logits.data(), s.x.data(), w.wcls, dim, p.vocab_size);
}

// ----------------------------------------------------------------------------
// 5. 采样逻辑 (Greedy / Top-P)
// ----------------------------------------------------------------------------

int sample_token(std::vector<float>& logits, float temperature = 0.85f, float top_p = 0.85f) {
    if (temperature == 0.0f) {
        // 贪心采样 (Greedy)
        int best_idx = 0;
        float best_val = logits[0];
        for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // 应用 Temperature
    for (float& l : logits) l /= temperature;
    softmax(logits.data(), logits.size());

    // 简化版带随机性的抽样
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        cdf += logits[i];
        if (r <= cdf) return i;
    }
    return logits.size() - 1;
}

// ----------------------------------------------------------------------------
// 6. 主程序 (加载模型 & 交互式对话)
// ----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string model_path = "models/minimind2.bin";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "====================================================\n";
    std::cout << " MiniMind2 C++ 原生推理引擎 (CPU/Single-File)\n";
    std::cout << " 正在加载模型二进制: " << model_path << "\n";
    std::cout << "====================================================\n";

    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) {
        std::cerr << "错误：无法打开模型文件: " << model_path << "\n请先使用 python scripts/Tools/export_cpp_bin.py 导出 .bin 文件！\n";
        return 1;
    }

    // 1. 读取 Config Header
    Config config;
    if (fread(&config, sizeof(Config), 1, f) != 1) {
        std::cerr << "错误：读取 Header 失败！\n";
        return 1;
    }

    std::cout << "参数规格: dim=" << config.dim << ", layers=" << config.n_layers
              << ", heads=" << config.n_heads << ", kv_heads=" << config.n_kv_heads
              << ", vocab=" << config.vocab_size << ", max_seq=" << config.seq_len << "\n";

    // 2. 读取 Tokenizer 词表
    Tokenizer tokenizer(config.vocab_size);
    for (int i = 0; i < config.vocab_size; ++i) {
        uint16_t raw_len = 0;
        if (fread(&raw_len, sizeof(uint16_t), 1, f) != 1) break;
        std::vector<char> raw_buf(raw_len + 1, 0);
        if (raw_len > 0) fread(raw_buf.data(), sizeof(char), raw_len, f);
        tokenizer.raw_vocab[i] = std::string(raw_buf.data(), raw_len);

        uint16_t dec_len = 0;
        if (fread(&dec_len, sizeof(uint16_t), 1, f) != 1) break;
        std::vector<char> dec_buf(dec_len + 1, 0);
        if (dec_len > 0) fread(dec_buf.data(), sizeof(char), dec_len, f);
        tokenizer.decode_vocab[i] = std::string(dec_buf.data(), dec_len);
    }

    // 3. 读取权重
    long weights_start = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, weights_start, SEEK_SET);

    size_t weights_bytes = file_size - weights_start;
    std::vector<float> weight_data(weights_bytes / sizeof(float));
    if (fread(weight_data.data(), sizeof(float), weight_data.size(), f) != weight_data.size()) {
        std::cerr << "错误：读取权重矩阵失败！\n";
        return 1;
    }
    fclose(f);

    // 映射指针到权重数据结构
    TransformerWeights w;
    float* ptr = weight_data.data();
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int n_layers = config.n_layers;
    int kv_dim = config.n_kv_heads * (dim / config.n_heads);

    w.token_embedding_table = ptr; ptr += (size_t)config.vocab_size * dim;
    w.layers.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        w.layers[i].rms_att_weight = ptr; ptr += dim;
        w.layers[i].wq = ptr; ptr += dim * dim;
        w.layers[i].wk = ptr; ptr += kv_dim * dim;
        w.layers[i].wv = ptr; ptr += kv_dim * dim;
        w.layers[i].wo = ptr; ptr += dim * dim;
        w.layers[i].rms_ffn_weight = ptr; ptr += dim;
        w.layers[i].w_gate = ptr; ptr += hidden_dim * dim;
        w.layers[i].w_up = ptr; ptr += hidden_dim * dim;
        w.layers[i].w_down = ptr; ptr += dim * hidden_dim;
    }
    w.rms_final_weight = ptr; ptr += dim;
    w.wcls = ptr; ptr += (size_t)config.vocab_size * dim;

    RunState state(config);
    std::cout << "✅ 模型加载完毕！随时可以在下方输入问题开始对话（输入 exit 退出）。\n\n";

    // 4. 交互式聊天循环
    while (true) {
        std::cout << "User > ";
        std::string user_input;
        if (!std::getline(std::cin, user_input) || user_input == "exit") {
            break;
        }
        if (user_input.empty()) continue;

        // 构造 MiniMind ChatML 提示词模板
        std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";

        std::vector<int> prompt_tokens = tokenizer.encode(prompt);

        std::cout << "MiniMind > " << std::flush;
        auto start_t = std::chrono::high_resolution_clock::now();

        int token = 1; // 默认 BOS / 起始
        int pos = 0;
        int max_gen = 256;
        int generated_count = 0;

        // 4.1 Prefill 提示词 Token
        for (; pos < (int)prompt_tokens.size(); ++pos) {
            token = prompt_tokens[pos];
            forward(config, w, state, token, pos);
        }

        // 4.2 Decode 自回归生成循环
        for (int step = 0; step < max_gen && pos < config.seq_len - 1; ++step, ++pos) {
            int next_token = sample_token(state.logits, 0.85f, 0.85f);
            if (next_token == 2 || next_token == 0) { // EOS 结束符
                break;
            }
            std::string piece = tokenizer.decode(next_token);
            std::cout << piece << std::flush;
            generated_count++;

            forward(config, w, state, next_token, pos);
        }

        auto end_t = std::chrono::high_resolution_clock::now();
        double cost_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();
        std::cout << "\n[" << generated_count << " tokens, " << (generated_count / (cost_ms / 1000.0)) << " tok/s]\n\n";
    }

    std::cout << "再见！\n";
    return 0;
}
