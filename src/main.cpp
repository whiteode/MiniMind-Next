#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "model/model.h"
#include "model/embedding.h"
#include "tokenizer/mmap_tokenizer.h"
#include "sampler/sampler.h"

using namespace minimind;

int main(int argc, char* argv[]) {
    std::string model_path = "models/minimind2.bin";
    std::string emb_path = "models/embedding/embedding_fp16.embedding";
    std::string vocab_path = "models/minimind2.vocab.bin";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) emb_path = argv[2];
    if (argc > 3) vocab_path = argv[3];

    std::cout << "====================================================\n";
    std::cout << " MiniMind2 C++ 原生推理引擎 (CPU/Modular/Quantized)\n";
    std::cout << " Transformer 权重: " << model_path << "\n";
    std::cout << " Embedding 矩阵:   " << emb_path << "\n";
    std::cout << " MMap 二进制词表:  " << vocab_path << "\n";
    std::cout << "====================================================\n";

    // 1. 加载 Tokenizer
    MMapTokenizer tokenizer;
    if (!tokenizer.load(vocab_path)) {
        std::cerr << "错误：加载词表失败: " << vocab_path
                  << "\n请先运行: python scripts/Tools/export_tokenizer_bin.py\n";
        return 1;
    }

    // 2. 加载 Embedding 矩阵 (支持 FP16 / INT8 / INT4 零拷贝 mmap 动态反量化)
    QuantizedEmbedding embedding;
    if (!embedding.load(emb_path, LoadMode::DISK)) {
        std::cerr << "错误：加载 Embedding 失败: " << emb_path
                  << "\n请先运行: python scripts/Tools/export_embedding.py\n";
        return 1;
    }

    // 3. 打开模型权重文件并读取 Header
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) {
        std::cerr << "错误：无法打开模型文件: " << model_path
                  << "\n请先运行: python scripts/Tools/export_cpp_bin.py\n";
        return 1;
    }

    ModelConfig config;
    if (fread(&config, sizeof(ModelConfig), 1, f) != 1) {
        std::cerr << "错误：读取 Header 失败！\n";
        fclose(f);
        return 1;
    }

    std::cout << "模型规格: dim=" << config.dim << ", layers=" << config.n_layers
              << ", heads=" << config.n_heads << ", kv_heads=" << config.n_kv_heads
              << ", vocab=" << config.vocab_size << ", max_seq=" << config.seq_len << "\n";

    // 4. 加载 Transformer 层权重
    MiniMindModel model;
    model.config = config;
    if (!model.load_weights(f)) {
        std::cerr << "错误：读取权重失败！\n";
        fclose(f);
        return 1;
    }
    fclose(f);

    // 5. 初始化运行缓存与采样器
    RunState state(config);
    Sampler sampler; // 默认 temperature=0.85, top_p=0.85

    std::cout << "✅ 模型与组件加载完毕！随时可以在下方输入问题开始对话（输入 exit 退出）。\n\n";

    // 6. 交互式终端对话循环
    while (true) {
        std::cout << "User > ";
        std::string user_input;
        if (!std::getline(std::cin, user_input) || user_input == "exit") {
            break;
        }
        if (user_input.empty()) continue;

        // 构建 ChatML 模板并编码
        std::string prompt = tokenizer.apply_chat_template(user_input);
        std::vector<int> prompt_tokens = tokenizer.encode(prompt);

        std::cout << "MiniMind > " << std::flush;

        int token = 1;
        int pos = 0;
        int max_gen = 256;
        int generated_count = 0;
        std::string stream_buffer;

        // 6.1 Prefill 提示词 Token 测时
        auto prefill_start_t = std::chrono::high_resolution_clock::now();
        for (; pos < (int)prompt_tokens.size(); ++pos) {
            token = prompt_tokens[pos];
            model.forward(state, embedding, token, pos);
        }
        auto prefill_end_t = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end_t - prefill_start_t).count();

        // 6.2 Decode 自回归逐 Token 生成测时
        auto decode_start_t = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < max_gen && pos < config.seq_len - 1; ++step, ++pos) {
            int next_token = sampler.sample(state.logits);
            if (next_token == 2 || next_token == 0) { // EOS 结束符
                break;
            }
            std::string piece = tokenizer.decode_stream(next_token, stream_buffer);
            if (!piece.empty()) {
                std::cout << piece << std::flush;
            }
            generated_count++;

            model.forward(state, embedding, next_token, pos);
        }

        // 刷新生成结束时 buffer 中剩余的字节
        if (!stream_buffer.empty()) {
            std::cout << stream_buffer << std::flush;
        }

        auto decode_end_t = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(decode_end_t - decode_start_t).count();
        double decode_tok_per_sec = generated_count > 0 ? (generated_count / (decode_ms / 1000.0)) : 0.0;
        double prefill_tok_per_sec = prompt_tokens.size() > 0 ? (prompt_tokens.size() / (prefill_ms / 1000.0)) : 0.0;

        std::cout << "\n[Prefill: " << prompt_tokens.size() << " tokens in " << prefill_ms << " ms (" << prefill_tok_per_sec << " tok/s) | "
                  << "Decode: " << generated_count << " tokens in " << decode_ms << " ms (" << decode_tok_per_sec << " tok/s)]\n\n";
    }

    std::cout << "再见！\n";
    return 0;
}
