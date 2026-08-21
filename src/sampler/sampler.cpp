#include "sampler.h"
#include "../ops/math_ops.h"

namespace minimind {

Sampler::Sampler(const SamplerConfig& cfg) : config(cfg) {}

int Sampler::sample(std::vector<float>& logits) const {
    if (config.temperature == 0.0f) {
        // 贪心采样 (Greedy)
        int best_idx = 0;
        float best_val = logits[0];
        for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // 1. 应用 Temperature 缩放并做 Softmax
    for (float& l : logits) l /= config.temperature;
    ops::softmax(logits.data(), logits.size());

    // 2. Top-P (Nucleus) 截断采样
    if (config.top_p > 0.0f && config.top_p < 1.0f) {
        std::vector<std::pair<float, int>> probs;
        probs.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            probs.push_back({logits[i], (int)i});
        }
        std::sort(probs.rbegin(), probs.rend());

        float cumsum = 0.0f;
        int cutoff_idx = (int)probs.size() - 1;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i].first;
            if (cumsum >= config.top_p) {
                cutoff_idx = (int)i;
                break;
            }
        }

        float r = (float)rand() / (float)RAND_MAX * cumsum;
        float acc = 0.0f;
        for (int i = 0; i <= cutoff_idx; ++i) {
            acc += probs[i].first;
            if (r <= acc) return probs[i].second;
        }
        return probs[0].second;
    }

    // 3. 普通随机采样
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        cdf += logits[i];
        if (r <= cdf) return (int)i;
    }
    return (int)logits.size() - 1;
}

} // namespace minimind
