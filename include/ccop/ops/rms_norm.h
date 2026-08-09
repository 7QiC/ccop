#pragma once

#include <cstdint>

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// RMSNorm (Qwen3/LLaMA style):
//
//   rms = sqrt(mean_j x[i,j]^2 + eps)
//   out[i,j] = x[i,j] / rms * w[j]
//
// input:  [rows, dim] 2D contiguous view (activation)
// weight: [dim] 1D view
// out:    [rows, dim], same dtype as input
//
// This is the first mixed-dtype operator: activation and weight may have
// different dtypes (e.g. BF16 activation + FP32 weight), so dispatch is a
// combination of both Tensor dtypes.
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方检查。
// 第一阶段：先正确实现，不做性能优化。
// -----------------------------------------------------------------------------
void rms_norm(Tensor* out, const Tensor& input, const Tensor& weight, float eps,
              const ExecutionContext& ctx);

}  // namespace ccop
