#include "math_ops.h"

namespace minimind {
namespace ops {

void rmsnorm(float* o, const float* x, const float* weight, int size, float eps) {
    float ss = 0.0f;
    for (int j = 0; j < size; ++j) ss += x[j] * x[j];
    ss /= size;
    ss += eps;
    float rsqrt_ss = 1.0f / std::sqrt(ss);
    for (int j = 0; j < size; ++j) {
        o[j] = weight[j] * (rsqrt_ss * x[j]);
    }
}

void matmul(float* xout, const float* x, const float* w, int n, int d) {
    // xout: [d], x: [n], w: [d, n] (行优先存储)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        float val = 0.0f;
        const float* w_row = w + i * n;
        for (int j = 0; j < n; ++j) {
            val += w_row[j] * x[j];
        }
        xout[i] = val;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

void apply_rope(float* vec, int head_dim, int pos, float rope_theta) {
    int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / std::pow(rope_theta, (float)(i * 2) / (float)head_dim);
        float val = pos * freq;
        float fcr = std::cos(val);
        float fci = std::sin(val);
        float v0 = vec[i];
        float v1 = vec[i + half_dim];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + half_dim] = v0 * fci + v1 * fcr;
    }
}

} // namespace ops
} // namespace minimind
