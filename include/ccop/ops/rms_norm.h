#pragma once

#include <cstdint>

#include "ccop/error.h"
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
// different dtypes, so dispatch is a combination of both Tensor dtypes.
// Supported combinations: BF16×FP32, BF16×BF16, FP32×FP32.
//
// 算子只负责计算：参数错误返回 ErrorCode；kernel 错误映射后返回 ErrorCode。
// 第一阶段：先正确实现，不做性能优化。
// -----------------------------------------------------------------------------
Result<void> rms_norm(Tensor* out, const Tensor& input, const Tensor& weight, float eps,
                      const ExecutionContext& ctx);

}  // namespace ccop
