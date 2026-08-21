#pragma once

#include <vector>
#include <cstdlib>
#include <algorithm>

namespace minimind {

struct SamplerConfig {
    float temperature = 0.85f;
    float top_p = 0.85f;
};

class Sampler {
public:
    SamplerConfig config;

    explicit Sampler(const SamplerConfig& cfg = SamplerConfig());

    // 根据输入 logits 执行 Temperature 与 Top-P 采样，返回选中的 Token ID
    int sample(std::vector<float>& logits) const;
};

} // namespace minimind
