#pragma once

#include <cstdint>

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Row-wise softmax（attention 核心组件）：
//
//   input: [rows, cols] 2D 连续视图
//   out:   [rows, cols]，与 input 同 dtype（独立输出数组）
//
// 数学形式（逐行，数值稳定版）：
//   m_i    = max_j x[i][j]
//   out[i][j] = exp(x[i][j] - m_i) / sum_j exp(x[i][j] - m_i)
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方检查。
// 第一阶段：scalar 正确版（每线程一行），不做性能优化。
// -----------------------------------------------------------------------------
void softmax(Tensor* out, const Tensor& input, const ExecutionContext& ctx);

}  // namespace ccop
