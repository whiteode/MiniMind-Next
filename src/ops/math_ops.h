#pragma once

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace minimind {
namespace ops {

// 均方根归一化 (RMSNorm)
void rmsnorm(float* o, const float* x, const float* weight, int size, float eps = 1e-5f);

// 矩阵向量乘法 (xout = W * x)，OpenMP 并行加速
void matmul(float* xout, const float* x, const float* w, int n, int d);

// Softmax 归一化
void softmax(float* x, int size);

// RoPE 旋转位置编码 (对齐 HuggingFace rotate_half 方案)
void apply_rope(float* vec, int head_dim, int pos, float rope_theta = 1000000.0f);

} // namespace ops
} // namespace minimind
