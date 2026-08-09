#pragma once

#include <cstdint>

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Row-wise sum reduction, the only reduction shape LLM serving needs
// (base primitive for RMSNorm / softmax denominators).
//
// input:  [rows, cols] 2D contiguous view
// out:    [rows] 1D view, out[i] = sum_j input[i * cols + j]
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方（框架
// Backend）通过 cudaGetLastError / 同步检查。
// 第一阶段：先正确实现，不做性能优化（coalescing/tiling 后续）。
// -----------------------------------------------------------------------------
void reduce_sum_rows(Tensor* out, const Tensor& input, const ExecutionContext& ctx);

}  // namespace ccop
