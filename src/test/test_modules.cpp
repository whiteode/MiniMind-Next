#include "model/embedding.h"
#include "tokenizer/mmap_tokenizer.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace minimind;

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "🧪 MiniMind Quantized Embedding & Tokenizer" << std::endl;
    std::cout << "==========================================" << std::endl;

    // 1. 测试 Tokenizer
    std::string vocab_path = "models/minimind2.vocab.bin";
    MMapTokenizer tokenizer;
    if (tokenizer.load(vocab_path)) {
        std::cout << "✅ 词表加载成功! Vocab Size: " << tokenizer.get_vocab_size() << std::endl;

        std::string test_prompt = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n";
        auto tokens = tokenizer.encode(test_prompt);
        std::cout << "Prompt: " << test_prompt << std::endl;
        std::cout << "Encoded Tokens: [ ";
        for (int tid : tokens) std::cout << tid << " ";
        std::cout << "]" << std::endl;

        std::string decoded = tokenizer.decode(tokens);
        std::cout << "Decoded String: " << decoded << std::endl;
        if (decoded == test_prompt) {
            std::cout << "✅ Tokenizer Encode -> Decode 完美可逆对齐!" << std::endl;
        } else {
            std::cout << "⚠️ Tokenizer Decode 存在差异!" << std::endl;
        }
    } else {
        std::cout << "⚠️ 未找到 " << vocab_path << "，跳过 Tokenizer 单测" << std::endl;
    }

    std::cout << "------------------------------------------" << std::endl;

    // 2. 测试 Embedding (FP16 / INT8 / INT4)
    std::vector<std::string> emb_files = {
        "models/embedding/embedding_fp16.embedding",
        "models/embedding/embedding_int8_token.embedding",
        "models/embedding/embedding_int4_group.embedding"
    };

    for (const auto& path : emb_files) {
        QuantizedEmbedding emb;
        if (emb.load(path, LoadMode::DISK)) {
            std::cout << "✅ 加载成功: " << path << std::endl;
            std::cout << "   - Vocab: " << emb.get_vocab_size()
                      << " | Dim: " << emb.get_embedding_dim()
                      << " | Group: " << emb.get_group_size()
                      << " | Type: " << static_cast<int>(emb.get_quant_type()) << std::endl;

            auto vec = emb.forward(100);
            float sum = 0.0f;
            for (float v : vec) sum += v * v;
            float norm = std::sqrt(sum);
            std::cout << "   - Token ID 100 Embedding L2-Norm: " << std::fixed << std::setprecision(4) << norm << std::endl;
        } else {
            std::cout << "⚠️ 加载失败或未找到: " << path << std::endl;
        }
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "🎉 单元测试执行完毕!" << std::endl;
    return 0;
}
