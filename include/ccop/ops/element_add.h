#pragma once

#include "ccop/error.h"
#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// 逐元素原地加法（残差连接：dst += src）：
//
//   dst: [n]（原地修改）
//   src: [n]（只读）
//
// 数学形式：
//   dst[i] ← dst[i] + src[i]
//
// dst 与 src 必须同 dtype（BF16 或 FP32）、一维连续且长度一致；
// dst 与 src 为不同数组（不同 buffer）。
//
// 算子只负责计算：参数错误返回 ErrorCode；kernel 错误映射后返回 ErrorCode。
// 第一阶段：scalar 正确版，不做性能优化。
// -----------------------------------------------------------------------------
Result<void> element_add(Tensor* dst, const Tensor& src, const ExecutionContext& ctx);

}  // namespace ccop
