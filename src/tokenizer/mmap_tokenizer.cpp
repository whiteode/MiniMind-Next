#include "tokenizer/mmap_tokenizer.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <queue>
#include <iostream>

namespace minimind {

namespace {
    const std::vector<std::string> SPECIAL_TOKENS = {
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>"
    };

    struct TextSegment {
        std::string_view text;
        bool is_special;
        int token_id;
    };

    struct MergeQueueItem {
        int rank;
        int left_idx;
        int right_idx;

        bool operator>(const MergeQueueItem& other) const {
            if (rank != other.rank) {
                return rank > other.rank;
            }
            return left_idx > other.left_idx;
        }
    };

    struct SymbolNode {
        int prev;
        int next;
        int pos;
        int len;
        int id;
        bool active;
    };

    std::vector<TextSegment> split_special_tokens(std::string_view text, const MMapTokenizer* tokenizer) {
        std::vector<TextSegment> segments;
        size_t pos = 0;

        while (pos < text.length()) {
            size_t min_pos = std::string::npos;
            std::string_view found_token;

            for (const auto& special : SPECIAL_TOKENS) {
                size_t found = text.find(special, pos);
                if (found != std::string::npos && (min_pos == std::string::npos || found < min_pos)) {
                    min_pos = found;
                    found_token = special;
                }
            }

            if (min_pos == std::string::npos) {
                if (pos < text.length()) {
                    segments.push_back({text.substr(pos), false, -1});
                }
                break;
            }

            if (min_pos > pos) {
                segments.push_back({text.substr(pos, min_pos - pos), false, -1});
            }

            int token_id = tokenizer->find_exact_token(found_token);
            segments.push_back({found_token, true, token_id});
            pos = min_pos + found_token.length();
        }

        return segments;
    }
}

MMapTokenizer::~MMapTokenizer() {
    release_mmap();
}

MMapTokenizer::MMapTokenizer(MMapTokenizer&& other) noexcept {
    *this = std::move(other);
}

MMapTokenizer& MMapTokenizer::operator=(MMapTokenizer&& other) noexcept {
    if (this != &other) {
        release_mmap();
        fd_ = other.fd_;
        file_size_ = other.file_size_;
        mmap_ptr_ = other.mmap_ptr_;
        n_tokens_ = other.n_tokens_;
        n_sorted_ = other.n_sorted_;
        info_array_ = other.info_array_;
        sorted_ids_ = other.sorted_ids_;
        string_data_ = other.string_data_;
        bpe_ranks_ = std::move(other.bpe_ranks_);
        byte_to_token_ = other.byte_to_token_;

        other.fd_ = -1;
        other.file_size_ = 0;
        other.mmap_ptr_ = nullptr;
    }
    return *this;
}

void MMapTokenizer::release_mmap() {
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
        munmap(const_cast<char*>(mmap_ptr_), file_size_);
        mmap_ptr_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

bool MMapTokenizer::load(const std::string& filepath) {
    release_mmap();

    fd_ = open(filepath.c_str(), O_RDONLY);
    if (fd_ == -1) {
        std::cerr << "无法打开词表文件: " << filepath << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        std::cerr << "无法获取词表文件状态: " << filepath << std::endl;
        close(fd_);
        fd_ = -1;
        return false;
    }
    file_size_ = sb.st_size;

    mmap_ptr_ = static_cast<const char*>(mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0));
    if (mmap_ptr_ == MAP_FAILED) {
        std::cerr << "mmap 映射词表失败！" << std::endl;
        release_mmap();
        return false;
    }

    const char* curr_ptr = mmap_ptr_;

    n_tokens_ = *reinterpret_cast<const uint32_t*>(curr_ptr);
    curr_ptr += sizeof(uint32_t);

    n_sorted_ = *reinterpret_cast<const uint32_t*>(curr_ptr);
    curr_ptr += sizeof(uint32_t);

    info_array_ = reinterpret_cast<const BinaryTokenInfo*>(curr_ptr);
    curr_ptr += n_tokens_ * sizeof(BinaryTokenInfo);

    sorted_ids_ = reinterpret_cast<const uint32_t*>(curr_ptr);
    curr_ptr += n_sorted_ * sizeof(uint32_t);

    string_data_ = curr_ptr;

    size_t total_string_size = 0;
    for (uint32_t i = 0; i < n_tokens_; ++i) {
        total_string_size = std::max(total_string_size, static_cast<size_t>(info_array_[i].offset + info_array_[i].length));
    }
    curr_ptr += total_string_size;

    bpe_ranks_.clear();
    byte_to_token_ = nullptr;

    if (curr_ptr < mmap_ptr_ + file_size_) {
        uint32_t n_merges = *reinterpret_cast<const uint32_t*>(curr_ptr);
        curr_ptr += sizeof(uint32_t);

        const uint32_t* merges_data = reinterpret_cast<const uint32_t*>(curr_ptr);
        for (uint32_t i = 0; i < n_merges; ++i) {
            uint32_t left_id = merges_data[i * 3];
            uint32_t right_id = merges_data[i * 3 + 1];
            uint32_t rank = merges_data[i * 3 + 2];
            bpe_ranks_[{left_id, right_id}] = static_cast<int>(rank);
        }
        curr_ptr += n_merges * 3 * sizeof(uint32_t);

        if (curr_ptr + 256 * sizeof(uint32_t) <= mmap_ptr_ + file_size_) {
            byte_to_token_ = reinterpret_cast<const uint32_t*>(curr_ptr);
        }
    }

    return true;
}

std::string_view MMapTokenizer::decode_view(uint32_t token_id) const {
    if (token_id >= n_tokens_) return "";
    const BinaryTokenInfo& info = info_array_[token_id];
    return std::string_view(string_data_ + info.offset, info.length);
}

int MMapTokenizer::find_exact_token(std::string_view text) const {
    if (!sorted_ids_) return -1;
    auto it = std::lower_bound(
        sorted_ids_, sorted_ids_ + n_sorted_, text,
        [this](uint32_t tid, std::string_view target) {
            return this->decode_view(tid) < target;
        }
    );

    if (it != sorted_ids_ + n_sorted_) {
        if (this->decode_view(*it) == text) {
            return static_cast<int>(*it);
        }
    }
    return -1;
}

std::string MMapTokenizer::decode(int token_id) const {
    if (token_id < 0 || static_cast<uint32_t>(token_id) >= n_tokens_) return "";
    return std::string(decode_view(static_cast<uint32_t>(token_id)));
}

// 辅助函数：根据首字节计算 UTF-8 字符所需的总字节长度
static int get_utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;          // 0xxxxxxx (ASCII, 1 字节)
    if ((c & 0xE0) == 0xC0) return 2;       // 110xxxxx (2 字节)
    if ((c & 0xF0) == 0xE0) return 3;       // 1110xxxx (常用中文汉字, 3 字节)
    if ((c & 0xF8) == 0xF0) return 4;       // 11110xxx (Emoji/特殊符, 4 字节)
    return 1; // 异常字节兜底
}

std::string MMapTokenizer::decode_stream(int token_id, std::string& buffer) const {
    std::string piece = decode(token_id);
    if (piece.empty()) return "";

    buffer += piece;

    size_t valid_bytes = 0;
    size_t i = 0;
    while (i < buffer.size()) {
        unsigned char c = static_cast<unsigned char>(buffer[i]);
        int char_len = get_utf8_char_len(c);

        if (i + char_len <= buffer.size()) {
            valid_bytes = i + char_len;
            i += char_len;
        } else {
            // 遇到了尚未拼凑完整的 UTF-8 字符边界，暂留在 buffer 中等待后续 Token
            break;
        }
    }

    if (valid_bytes > 0) {
        std::string output = buffer.substr(0, valid_bytes);
        buffer.erase(0, valid_bytes);
        return output;
    }

    return "";
}

std::string MMapTokenizer::apply_chat_template(const std::string& user_query, const std::string& system_prompt) const {
    return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
           "<|im_start|>user\n" + user_query + "<|im_end|>\n" +
           "<|im_start|>assistant\n";
}

std::string MMapTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int tid : tokens) {
        result += decode(tid);
    }
    return result;
}

std::vector<int> MMapTokenizer::encode(const std::string& text) const {
    return encode_mmap(text);
}

std::vector<int> MMapTokenizer::encode_mmap(std::string_view text) const {
    if (text.empty()) return {};
    if (!byte_to_token_) {
        std::cerr << "错误: byte_to_token 未加载" << std::endl;
        return {};
    }

    auto segments = split_special_tokens(text, this);
    std::vector<int> result;
    result.reserve(text.length() / 2);

    for (const auto& segment : segments) {
        if (segment.is_special) {
            if (segment.token_id != -1) {
                result.push_back(segment.token_id);
            }
        } else {
            auto tokens = encode_bpe_segment(segment.text);
            result.insert(result.end(), tokens.begin(), tokens.end());
        }
    }

    return result;
}

std::vector<int> MMapTokenizer::encode_bpe_segment(std::string_view text) const {
    if (text.empty()) return {};

    int n = static_cast<int>(text.length());
    std::vector<SymbolNode> nodes(n);
    for (int i = 0; i < n; ++i) {
        unsigned char byte_val = static_cast<unsigned char>(text[i]);
        nodes[i] = {
            i - 1,
            i + 1,
            i,
            1,
            byte_to_token_ ? static_cast<int>(byte_to_token_[byte_val]) : -1,
            true
        };
    }
    nodes[n - 1].next = -1;

    std::priority_queue<MergeQueueItem, std::vector<MergeQueueItem>, std::greater<MergeQueueItem>> pq;

    auto try_add_merge = [&](int left, int right) {
        if (left == -1 || right == -1) return;
        int left_id = nodes[left].id;
        int right_id = nodes[right].id;
        if (left_id == -1 || right_id == -1) return;

        auto it = bpe_ranks_.find({static_cast<uint32_t>(left_id), static_cast<uint32_t>(right_id)});
        if (it != bpe_ranks_.end()) {
            pq.push({it->second, left, right});
        }
    };

    for (int i = 0; i < n - 1; ++i) {
        try_add_merge(i, i + 1);
    }

    while (!pq.empty()) {
        auto top = pq.top();
        pq.pop();

        int left = top.left_idx;
        int right = top.right_idx;

        if (!nodes[left].active || !nodes[right].active || nodes[left].next != right) {
            continue;
        }

        std::string_view merged_str(text.data() + nodes[left].pos, nodes[left].len + nodes[right].len);
        int merged_id = find_exact_token(merged_str);

        nodes[left].len += nodes[right].len;
        nodes[left].id = merged_id;
        nodes[right].active = false;

        nodes[left].next = nodes[right].next;
        if (nodes[right].next != -1) {
            nodes[nodes[right].next].prev = left;
        }

        try_add_merge(nodes[left].prev, left);
        try_add_merge(left, nodes[left].next);
    }

    std::vector<int> tokens;
    tokens.reserve(n / 2);

    int curr = 0;
    while (curr != -1) {
        if (nodes[curr].id != -1) {
            tokens.push_back(nodes[curr].id);
        }
        curr = nodes[curr].next;
    }

    return tokens;
}

} // namespace minimind
