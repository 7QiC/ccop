#pragma once

#include <cstdint>

#include "ccop/error.h"
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
// 算子只负责计算：参数错误返回 ErrorCode；kernel 错误映射后返回 ErrorCode。
// 第一阶段：scalar 正确版（每线程一行），不做性能优化。
// -----------------------------------------------------------------------------
Result<void> softmax(Tensor* out, const Tensor& input, const ExecutionContext& ctx);

}  // namespace ccop
