#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>

namespace minimind {

class Tokenizer {
public:
    int vocab_size;
    std::vector<std::string> vocab;

    Tokenizer();
    explicit Tokenizer(int v_size);

    // 从打开的文件描述符中载入词表
    bool load(FILE* f, int v_size);

    // 字符串编码为 Token IDs (最长匹配)
    std::vector<int> encode(const std::string& text) const;

    // Token ID 解码为对应原始字节串
    std::string decode(int token_id) const;

    // 流式解码：处理多 Token 拼凑单个多字节 UTF-8 字符（汉字切分）导致的乱码，仅返回已拼装完整的 UTF-8 文本
    std::string decode_stream(int token_id, std::string& buffer) const;

    // 构建 ChatML 格式的对话提示词
    std::string apply_chat_template(const std::string& user_query, const std::string& system_prompt = "You are a helpful assistant.") const;
};

} // namespace minimind
