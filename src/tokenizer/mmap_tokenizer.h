#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace minimind {

#pragma pack(push, 1)
struct BinaryTokenInfo {
    uint32_t offset;
    uint32_t length;
};
#pragma pack(pop)

class MMapTokenizer {
public:
    MMapTokenizer() = default;
    ~MMapTokenizer();

    MMapTokenizer(const MMapTokenizer&) = delete;
    MMapTokenizer& operator=(const MMapTokenizer&) = delete;
    MMapTokenizer(MMapTokenizer&& other) noexcept;
    MMapTokenizer& operator=(MMapTokenizer&& other) noexcept;

    bool load(const std::string& filepath);

    // 编码与解码接口
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode(int token_id) const;

    // 零拷贝视图与工具方法
    std::string_view decode_view(uint32_t token_id) const;
    int find_exact_token(std::string_view text) const;
    uint32_t get_vocab_size() const { return n_tokens_; }

private:
    void release_mmap();
    std::vector<int> encode_mmap(std::string_view text) const;
    std::vector<int> encode_bpe_segment(std::string_view text) const;

    int fd_ = -1;
    size_t file_size_ = 0;
    const char* mmap_ptr_ = nullptr;

    uint32_t n_tokens_ = 0;
    uint32_t n_sorted_ = 0;
    const BinaryTokenInfo* info_array_ = nullptr;
    const uint32_t* sorted_ids_ = nullptr;
    const char* string_data_ = nullptr;
    std::map<std::pair<uint32_t, uint32_t>, int> bpe_ranks_;
    const uint32_t* byte_to_token_ = nullptr;
};

} // namespace minimind
