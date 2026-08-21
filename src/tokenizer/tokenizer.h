#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>

namespace minimind {

class Tokenizer {
public:
    int vocab_size;
    std::vector<std::string> raw_vocab;
    std::vector<std::string> decode_vocab;

    Tokenizer();
    explicit Tokenizer(int v_size);

    // 从打开的文件描述符中载入词表
    bool load(FILE* f, int v_size);

    // 字符串编码为 Token IDs (最长匹配)
    std::vector<int> encode(const std::string& text) const;

    // Token ID 解码为对应 UTF-8 字符串
    std::string decode(int token_id) const;

    // 构建 ChatML 格式的对话提示词
    std::string apply_chat_template(const std::string& user_query, const std::string& system_prompt = "You are a helpful assistant.") const;
};

} // namespace minimind
