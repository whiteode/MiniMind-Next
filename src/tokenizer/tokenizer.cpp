#include "tokenizer.h"
#include <iostream>

namespace minimind {

Tokenizer::Tokenizer() : vocab_size(0) {}

Tokenizer::Tokenizer(int v_size) : vocab_size(v_size), raw_vocab(v_size), decode_vocab(v_size) {}

bool Tokenizer::load(FILE* f, int v_size) {
    vocab_size = v_size;
    raw_vocab.resize(vocab_size);
    decode_vocab.resize(vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        uint16_t raw_len = 0;
        if (fread(&raw_len, sizeof(uint16_t), 1, f) != 1) return false;
        std::vector<char> raw_buf(raw_len + 1, 0);
        if (raw_len > 0) {
            if (fread(raw_buf.data(), sizeof(char), raw_len, f) != raw_len) return false;
        }
        raw_vocab[i] = std::string(raw_buf.data(), raw_len);

        uint16_t dec_len = 0;
        if (fread(&dec_len, sizeof(uint16_t), 1, f) != 1) return false;
        std::vector<char> dec_buf(dec_len + 1, 0);
        if (dec_len > 0) {
            if (fread(dec_buf.data(), sizeof(char), dec_len, f) != dec_len) return false;
        }
        decode_vocab[i] = std::string(dec_buf.data(), dec_len);
    }
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    size_t i = 0;
    while (i < text.size()) {
        int best_id = -1;
        size_t best_len = 0;
        // 匹配 decode_vocab 中最长前缀
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

std::string Tokenizer::decode(int token_id) const {
    if (token_id >= 0 && token_id < vocab_size) {
        return decode_vocab[token_id];
    }
    return "";
}

std::string Tokenizer::apply_chat_template(const std::string& user_query, const std::string& system_prompt) const {
    return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n" + user_query + "<|im_end|>\n<|im_start|>assistant\n";
}

} // namespace minimind
