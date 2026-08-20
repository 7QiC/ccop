#pragma once

#include <cstdint>

#include "ccop/error.h"
#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// Row-wise sum reduction, the only reduction shape LLM serving needs
// (base primitive for RMSNorm / softmax denominators).
//
// input:  [rows, cols] 2D contiguous view
// out:    [rows] 1D view, out[i] = sum_j input[i * cols + j]
// input/out 必须同 dtype（BF16 或 FP32）。
//
// 算子只负责计算：参数错误返回 ErrorCode；kernel 错误映射后返回 ErrorCode。
// 第一阶段：先正确实现，不做性能优化（coalescing/tiling 后续）。
Result<void> reduce_sum_rows(Tensor* out, const Tensor& input, const ExecutionContext& ctx);

}  // namespace ccop
