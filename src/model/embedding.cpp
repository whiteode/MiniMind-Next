#include "model/embedding.h"
#include <cstring>
#include <iostream>
#include <algorithm>

namespace minimind {

QuantizedEmbedding::QuantizedEmbedding(QuantizedEmbedding&& other) noexcept {
    *this = std::move(other);
}

QuantizedEmbedding& QuantizedEmbedding::operator=(QuantizedEmbedding&& other) noexcept {
    if (this != &other) {
        release_mmap();
        version_ = other.version_;
        quant_type_ = other.quant_type_;
        vocab_size_ = other.vocab_size_;
        embedding_dim_ = other.embedding_dim_;
        group_size_ = other.group_size_;
        mode_ = other.mode_;
        fd_ = other.fd_;
        mapped_size_ = other.mapped_size_;
        mapped_data_ = other.mapped_data_;
        weights_offset_ = other.weights_offset_;
        weight_bytes_per_token_ = other.weight_bytes_per_token_;
        weight_fp16_ = std::move(other.weight_fp16_);
        q_weight_int8_ = std::move(other.q_weight_int8_);
        q_weight_uint8_ = std::move(other.q_weight_uint8_);
        scales_ = std::move(other.scales_);
        zero_points_ = std::move(other.zero_points_);
        codebook_ = std::move(other.codebook_);

        other.fd_ = -1;
        other.mapped_data_ = nullptr;
        other.mapped_size_ = 0;
    }
    return *this;
}

void QuantizedEmbedding::release_mmap() {
    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
        munmap(mapped_data_, mapped_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

bool QuantizedEmbedding::load(const std::string& file_path, LoadMode mode) {
    mode_ = mode;
    release_mmap();

    fd_ = open(file_path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        std::cerr << "无法打开 Embedding 文件: " << file_path << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        std::cerr << "无法获取文件状态: " << file_path << std::endl;
        close(fd_);
        fd_ = -1;
        return false;
    }
    mapped_size_ = sb.st_size;

    mapped_data_ = static_cast<uint8_t*>(mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED, fd_, 0));
    if (mapped_data_ == MAP_FAILED) {
        std::cerr << "mmap 映射失败！" << std::endl;
        release_mmap();
        return false;
    }

    size_t offset = 0;
    if (std::string(reinterpret_cast<char*>(mapped_data_), 4) != "MMEM") {
        std::cerr << "非法的 .embedding 二进制文件格式 (Magic Word mismatch)!" << std::endl;
        release_mmap();
        return false;
    }
    offset += 4;

    auto read_val = [&](auto& val) {
        val = *reinterpret_cast<const std::remove_reference_t<decltype(val)>*>(mapped_data_ + offset);
        offset += sizeof(val);
    };

    uint32_t type_id;
    read_val(version_);
    read_val(type_id);
    read_val(vocab_size_);
    read_val(embedding_dim_);
    read_val(group_size_);
    quant_type_ = static_cast<QuantType>(type_id);

    size_t V = vocab_size_;
    size_t D = embedding_dim_;
    size_t G = (group_size_ > 0) ? (D / group_size_) : 0;

    weights_offset_ = offset;

    if (quant_type_ == QuantType::FP16) {
        weight_bytes_per_token_ = D * sizeof(uint16_t);
    } else if (quant_type_ == QuantType::INT8_TENSOR || quant_type_ == QuantType::INT8_TOKEN || quant_type_ == QuantType::INT8_GROUP ||
               quant_type_ == QuantType::UINT8_TENSOR || quant_type_ == QuantType::UINT8_TOKEN || quant_type_ == QuantType::UINT8_GROUP) {
        weight_bytes_per_token_ = D * 1;
    } else {
        weight_bytes_per_token_ = D / 2; // 4Bit 压缩
    }

    auto process_weights = [&](auto& vec, size_t count) {
        size_t byte_size = count * sizeof(typename std::remove_reference_t<decltype(vec)>::value_type);
        if (mode_ == LoadMode::MEMORY) {
            vec.resize(count);
            std::memcpy(vec.data(), mapped_data_ + offset, byte_size);
        }
        offset += byte_size;
    };

    auto read_aux_fp16 = [&](std::vector<float>& vec, size_t count) {
        vec.resize(count);
        const uint16_t* raw_fp16 = reinterpret_cast<const uint16_t*>(mapped_data_ + offset);
        for (size_t i = 0; i < count; ++i) {
            vec[i] = fp16_to_fp32(raw_fp16[i]);
        }
        offset += count * sizeof(uint16_t);
    };

    if (quant_type_ == QuantType::FP16) {
        process_weights(weight_fp16_, V * D);
    } else if (quant_type_ == QuantType::NF4_TENSOR) {
        process_weights(q_weight_uint8_, V * D / 2);
        read_aux_fp16(codebook_, 16);
    } else if (quant_type_ == QuantType::INT8_TENSOR || quant_type_ == QuantType::INT8_TOKEN || quant_type_ == QuantType::INT8_GROUP) {
        process_weights(q_weight_int8_, V * D);
        size_t scale_cnt = (quant_type_ == QuantType::INT8_TENSOR) ? 1 : ((quant_type_ == QuantType::INT8_TOKEN) ? V : V * G);
        read_aux_fp16(scales_, scale_cnt);
    } else if (quant_type_ == QuantType::UINT8_TENSOR || quant_type_ == QuantType::UINT8_TOKEN || quant_type_ == QuantType::UINT8_GROUP) {
        process_weights(q_weight_uint8_, V * D);
        size_t scale_cnt = (quant_type_ == QuantType::UINT8_TENSOR) ? 1 : ((quant_type_ == QuantType::UINT8_TOKEN) ? V : V * G);
        read_aux_fp16(scales_, scale_cnt);
        read_aux_fp16(zero_points_, scale_cnt);
    } else if (quant_type_ == QuantType::INT4_TENSOR || quant_type_ == QuantType::INT4_TOKEN || quant_type_ == QuantType::INT4_GROUP) {
        process_weights(q_weight_uint8_, V * D / 2);
        size_t scale_cnt = (quant_type_ == QuantType::INT4_TENSOR) ? 1 : ((quant_type_ == QuantType::INT4_TOKEN) ? V : V * G);
        read_aux_fp16(scales_, scale_cnt);
    } else if (quant_type_ == QuantType::UINT4_TENSOR || quant_type_ == QuantType::UINT4_TOKEN || quant_type_ == QuantType::UINT4_GROUP) {
        process_weights(q_weight_uint8_, V * D / 2);
        size_t scale_cnt = (quant_type_ == QuantType::UINT4_TENSOR) ? 1 : ((quant_type_ == QuantType::UINT4_TOKEN) ? V : V * G);
        read_aux_fp16(scales_, scale_cnt);
        read_aux_fp16(zero_points_, scale_cnt);
    } else {
        std::cerr << "不支持的 Embedding 量化类型: " << type_id << std::endl;
        release_mmap();
        return false;
    }

    if (mode_ == LoadMode::MEMORY) {
        release_mmap();
    }

    return true;
}

std::vector<float> QuantizedEmbedding::forward(int token_id) const {
    std::vector<float> output(embedding_dim_, 0.0f);
    if (token_id < 0 || token_id >= static_cast<int>(vocab_size_)) return output;

    const uint16_t* fp16_ptr  = nullptr;
    const int8_t* int8_ptr  = nullptr;
    const uint8_t* uint8_ptr = nullptr;

    if (mode_ == LoadMode::DISK) {
        size_t token_offset = weights_offset_ + static_cast<size_t>(token_id) * weight_bytes_per_token_;
        const uint8_t* target_ptr = mapped_data_ + token_offset;
        fp16_ptr  = reinterpret_cast<const uint16_t*>(target_ptr);
        int8_ptr  = reinterpret_cast<const int8_t*>(target_ptr);
        uint8_ptr = target_ptr;
    } else {
        if (!weight_fp16_.empty())    fp16_ptr  = weight_fp16_.data() + static_cast<size_t>(token_id) * embedding_dim_;
        if (!q_weight_int8_.empty())  int8_ptr  = q_weight_int8_.data() + static_cast<size_t>(token_id) * embedding_dim_;
        if (!q_weight_uint8_.empty()) uint8_ptr = q_weight_uint8_.data() + static_cast<size_t>(token_id) * weight_bytes_per_token_;
    }

    auto get_param_idx = [&](uint32_t dim_idx, QuantType t_tensor, QuantType t_token) -> size_t {
        if (quant_type_ == t_tensor) return 0;
        if (quant_type_ == t_token) return token_id;
        return static_cast<size_t>(token_id) * (embedding_dim_ / group_size_) + (dim_idx / group_size_);
    };

    if (quant_type_ == QuantType::FP16) {
        for (uint32_t i = 0; i < embedding_dim_; ++i) output[i] = fp16_to_fp32(fp16_ptr[i]);
    } else if (quant_type_ == QuantType::INT8_TENSOR || quant_type_ == QuantType::INT8_TOKEN || quant_type_ == QuantType::INT8_GROUP) {
        for (uint32_t i = 0; i < embedding_dim_; ++i) {
            size_t p_idx = get_param_idx(i, QuantType::INT8_TENSOR, QuantType::INT8_TOKEN);
            output[i] = static_cast<float>(int8_ptr[i]) * scales_[p_idx];
        }
    } else if (quant_type_ == QuantType::UINT8_TENSOR || quant_type_ == QuantType::UINT8_TOKEN || quant_type_ == QuantType::UINT8_GROUP) {
        for (uint32_t i = 0; i < embedding_dim_; ++i) {
            size_t p_idx = get_param_idx(i, QuantType::UINT8_TENSOR, QuantType::UINT8_TOKEN);
            output[i] = (static_cast<float>(uint8_ptr[i]) - zero_points_[p_idx]) * scales_[p_idx];
        }
    } else if (quant_type_ == QuantType::INT4_TENSOR || quant_type_ == QuantType::INT4_TOKEN || quant_type_ == QuantType::INT4_GROUP) {
        for (uint32_t i = 0; i < embedding_dim_ / 2; ++i) {
            uint8_t packed_byte = uint8_ptr[i];
            int8_t high = (packed_byte >> 4) & 0xF;
            int8_t low = packed_byte & 0xF;
            if (high >= 8) high -= 16;
            if (low >= 8) low -= 16;
            size_t p_idx_h = get_param_idx(i * 2, QuantType::INT4_TENSOR, QuantType::INT4_TOKEN);
            size_t p_idx_l = get_param_idx(i * 2 + 1, QuantType::INT4_TENSOR, QuantType::INT4_TOKEN);
            output[i * 2]     = static_cast<float>(high) * scales_[p_idx_h];
            output[i * 2 + 1] = static_cast<float>(low)  * scales_[p_idx_l];
        }
    } else if (quant_type_ == QuantType::UINT4_TENSOR || quant_type_ == QuantType::UINT4_TOKEN || quant_type_ == QuantType::UINT4_GROUP) {
        for (uint32_t i = 0; i < embedding_dim_ / 2; ++i) {
            uint8_t packed_byte = uint8_ptr[i];
            uint8_t high = (packed_byte >> 4) & 0xF;
            uint8_t low = packed_byte & 0xF;
            size_t p_idx_h = get_param_idx(i * 2, QuantType::UINT4_TENSOR, QuantType::UINT4_TOKEN);
            size_t p_idx_l = get_param_idx(i * 2 + 1, QuantType::UINT4_TENSOR, QuantType::UINT4_TOKEN);
            output[i * 2]     = (static_cast<float>(high) - zero_points_[p_idx_h]) * scales_[p_idx_h];
            output[i * 2 + 1] = (static_cast<float>(low)  - zero_points_[p_idx_l]) * scales_[p_idx_l];
        }
    } else if (quant_type_ == QuantType::NF4_TENSOR) {
        for (uint32_t i = 0; i < embedding_dim_ / 2; ++i) {
            uint8_t packed_byte = uint8_ptr[i];
            uint8_t high = (packed_byte >> 4) & 0xF;
            uint8_t low = packed_byte & 0xF;
            output[i * 2]     = codebook_[high];
            output[i * 2 + 1] = codebook_[low];
        }
    }

    return output;
}

std::vector<float> QuantizedEmbedding::forward_batch(const std::vector<int>& tokens) const {
    std::vector<float> outputs(tokens.size() * embedding_dim_, 0.0f);
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<float> single_emb = forward(tokens[i]);
        std::memcpy(outputs.data() + i * embedding_dim_, single_emb.data(), embedding_dim_ * sizeof(float));
    }
    return outputs;
}

} // namespace minimind
