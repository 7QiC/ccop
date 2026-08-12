#pragma once

#include <cstdint>

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// SiLU 门控乘法（Qwen3/LLaMA FFN 的 gate_proj × up_proj 组合）：
//
//   gate: [n]
//   up:   [n]
//   out:  [n]（独立输出数组，不原地覆盖 gate/up）
//
// 数学形式：
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//   out[i]  = silu(gate[i]) * up[i]
//
// gate 与 up 必须同 dtype（BF16 或 FP32），out 与 gate 同 dtype；
// 三个数组均为一维连续视图，长度一致。
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方检查。
// 第一阶段：scalar 正确版，不做性能优化。
// -----------------------------------------------------------------------------
void silu_mul(Tensor* out, const Tensor& gate, const Tensor& up,
              const ExecutionContext& ctx);

}  // namespace ccop
