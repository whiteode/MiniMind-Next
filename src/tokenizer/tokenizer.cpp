#include "tokenizer.h"
#include <iostream>

namespace minimind {

Tokenizer::Tokenizer() : vocab_size(0) {}

Tokenizer::Tokenizer(int v_size) : vocab_size(v_size), vocab(v_size) {}

bool Tokenizer::load(FILE* f, int v_size) {
    vocab_size = v_size;
    vocab.resize(vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        uint16_t len = 0;
        if (fread(&len, sizeof(uint16_t), 1, f) != 1) return false;
        std::vector<char> buf(len + 1, 0);
        if (len > 0) {
            if (fread(buf.data(), sizeof(char), len, f) != len) return false;
        }
        vocab[i] = std::string(buf.data(), len);
    }
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    size_t i = 0;
    while (i < text.size()) {
        int best_id = -1;
        size_t best_len = 0;
        // 匹配 vocab 中最长前缀
        for (int id = 0; id < vocab_size; ++id) {
            const std::string& token = vocab[id];
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
        return vocab[token_id];
    }
    return "";
}

// 辅助函数：根据首字节计算 UTF-8 字符所需的总字节长度
static int get_utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;          // 0xxxxxxx (ASCII, 1 字节)
    if ((c & 0xE0) == 0xC0) return 2;       // 110xxxxx (2 字节)
    if ((c & 0xF0) == 0xE0) return 3;       // 1110xxxx (常用中文汉字, 3 字节)
    if ((c & 0xF8) == 0xF0) return 4;       // 11110xxx (Emoji/特殊符, 4 字节)
    return 1; // 异常字节兜底
}

std::string Tokenizer::decode_stream(int token_id, std::string& buffer) const {
    std::string piece = decode(token_id);
    if (piece.empty()) return "";

    buffer += piece;
    std::string output;
    size_t i = 0;
    while (i < buffer.size()) {
        unsigned char c = (unsigned char)buffer[i];
        int need_len = get_utf8_char_len(c);

        // 如果缓冲区剩下的字节数还不够组成一个完整字符，留在 buffer 留待下一个 token 拼接
        if (i + need_len > buffer.size()) {
            break;
        }

        // 校验后续字节是否都是合法的 10xxxxxx
        bool valid = true;
        for (int k = 1; k < need_len; ++k) {
            if (((unsigned char)buffer[i + k] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }

        if (valid) {
            output.append(buffer, i, need_len);
            i += need_len;
        } else {
            // 遇到非标准字节，跳过该单字节
            output.push_back(buffer[i]);
            i += 1;
        }
    }

    // 保留未完成拼接的尾部残缺字节
    buffer.erase(0, i);
    return output;
}

std::string Tokenizer::apply_chat_template(const std::string& user_query, const std::string& system_prompt) const {
    return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n" + user_query + "<|im_end|>\n<|im_start|>assistant\n";
}

} // namespace minimind
