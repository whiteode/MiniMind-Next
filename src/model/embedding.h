#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

namespace minimind {

// 通用 IEEE 754 half-precision float16 转换工具 (x86_64 / aarch64 跨平台无依赖)
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // Subnormal
            while (!(mant & 0x0400)) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x0400;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // Inf / NaN
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        // Normalized
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float val;
    std::memcpy(&val, &f, sizeof(float));
    return val;
}

enum class QuantType : uint32_t {
    FP16 = 0,
    NF4_TENSOR = 1,
    INT4_TENSOR = 2, INT4_TOKEN = 3, INT4_GROUP = 4,
    UINT4_TENSOR = 5, UINT4_TOKEN = 6, UINT4_GROUP = 7,
    INT8_TENSOR = 8, INT8_TOKEN = 9, INT8_GROUP = 10,
    UINT8_TENSOR = 11, UINT8_TOKEN = 12, UINT8_GROUP = 13,
    UNKNOWN = 999
};

enum class LoadMode {
    MEMORY = 0, // 全部加载至内存
    DISK = 1    // 使用 mmap 零拷贝映射 (推荐)
};

class QuantizedEmbedding {
public:
    QuantizedEmbedding() = default;
    ~QuantizedEmbedding() {
        release_mmap();
    }

    // 禁用拷贝，保证 mmap 资源安全
    QuantizedEmbedding(const QuantizedEmbedding&) = delete;
    QuantizedEmbedding& operator=(const QuantizedEmbedding&) = delete;
    QuantizedEmbedding(QuantizedEmbedding&& other) noexcept;
    QuantizedEmbedding& operator=(QuantizedEmbedding&& other) noexcept;

    bool load(const std::string& file_path, LoadMode mode = LoadMode::DISK);
    std::vector<float> forward(int token_id) const;
    std::vector<float> forward_batch(const std::vector<int>& tokens) const;

    uint32_t get_vocab_size() const { return vocab_size_; }
    uint32_t get_embedding_dim() const { return embedding_dim_; }
    uint32_t get_group_size() const { return group_size_; }
    QuantType get_quant_type() const { return quant_type_; }
    LoadMode get_load_mode() const { return mode_; }

private:
    void release_mmap();

    uint32_t version_ = 0;
    QuantType quant_type_ = QuantType::UNKNOWN;
    uint32_t vocab_size_ = 0;
    uint32_t embedding_dim_ = 0;
    uint32_t group_size_ = 0;

    LoadMode mode_ = LoadMode::DISK;

    int fd_ = -1;
    size_t mapped_size_ = 0;
    uint8_t* mapped_data_ = nullptr;
    size_t weights_offset_ = 0;
    size_t weight_bytes_per_token_ = 0;

    // MEMORY 模式缓存 (以 uint16_t 保存 raw fp16)
    std::vector<uint16_t> weight_fp16_;
    std::vector<int8_t>   q_weight_int8_;
    std::vector<uint8_t>  q_weight_uint8_;

    // 辅助参数 (Scales, Zero Points, Codebook 均转为 float 以获得极速计算)
    std::vector<float> scales_;
    std::vector<float> zero_points_;
    std::vector<float> codebook_;
};

} // namespace minimind
