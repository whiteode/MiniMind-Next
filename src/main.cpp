#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "model/model.h"
#include "tokenizer/tokenizer.h"
#include "sampler/sampler.h"

using namespace minimind;

int main(int argc, char* argv[]) {
    std::string model_path = "models/minimind2.bin";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "====================================================\n";
    std::cout << " MiniMind2 C++ 原生推理引擎 (CPU/Modular)\n";
    std::cout << " 正在加载模型二进制: " << model_path << "\n";
    std::cout << "====================================================\n";

    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) {
        std::cerr << "错误：无法打开模型文件: " << model_path
                  << "\n请先使用 python scripts/Tools/export_cpp_bin.py 导出 .bin 文件！\n";
        return 1;
    }

    // 1. 读取超参数 Header
    ModelConfig config;
    if (fread(&config, sizeof(ModelConfig), 1, f) != 1) {
        std::cerr << "错误：读取 Header 失败！\n";
        fclose(f);
        return 1;
    }

    std::cout << "参数规格: dim=" << config.dim << ", layers=" << config.n_layers
              << ", heads=" << config.n_heads << ", kv_heads=" << config.n_kv_heads
              << ", vocab=" << config.vocab_size << ", max_seq=" << config.seq_len << "\n";

    // 2. 加载 Tokenizer
    Tokenizer tokenizer;
    if (!tokenizer.load(f, config.vocab_size)) {
        std::cerr << "错误：加载词表失败！\n";
        fclose(f);
        return 1;
    }

    // 3. 加载模型权重
    MiniMindModel model;
    model.config = config;
    if (!model.load_weights(f)) {
        std::cerr << "错误：读取权重失败！\n";
        fclose(f);
        return 1;
    }
    fclose(f);

    // 4. 初始化运行缓存与采样器
    RunState state(config);
    Sampler sampler; // 默认 temperature=0.85, top_p=0.85

    std::cout << "✅ 模型加载完毕！随时可以在下方输入问题开始对话（输入 exit 退出）。\n\n";

    // 5. 交互式终端对话循环
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
        auto start_t = std::chrono::high_resolution_clock::now();

        int token = 1;
        int pos = 0;
        int max_gen = 256;
        int generated_count = 0;

        // 5.1 Prefill 提示词 Token
        for (; pos < (int)prompt_tokens.size(); ++pos) {
            token = prompt_tokens[pos];
            model.forward(state, token, pos);
        }

        // 5.2 Decode 自回归逐 Token 生成循环
        for (int step = 0; step < max_gen && pos < config.seq_len - 1; ++step, ++pos) {
            int next_token = sampler.sample(state.logits);
            if (next_token == 2 || next_token == 0) { // EOS 结束符
                break;
            }
            std::string piece = tokenizer.decode(next_token);
            std::cout << piece << std::flush;
            generated_count++;

            model.forward(state, next_token, pos);
        }

        auto end_t = std::chrono::high_resolution_clock::now();
        double cost_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();
        std::cout << "\n[" << generated_count << " tokens, " << (generated_count / (cost_ms / 1000.0)) << " tok/s]\n\n";
    }

    std::cout << "再见！\n";
    return 0;
}
