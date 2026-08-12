#pragma once

#include <cstdint>

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Split-half RoPE (Qwen3/LLaMA style)，原地修改 q/k：
//
//   q: [num_tokens, num_q_heads, head_dim]
//   k: [num_tokens, num_kv_heads, head_dim]（q_heads 与 kv_heads 可不同，GQA）
//   positions: [num_tokens] int32（device），每个 token 的绝对位置（查 cache 用）
//   rope_cache: [max_position, rotary_dim / 2, 2] float32（device），
//               cache[..., 0] = cos，cache[..., 1] = sin
//   rotary_dim: 偶数且 <= head_dim；只旋转前 rotary_dim 维，
//               rotary_dim..head_dim 保持原值（partial rotary）
//
// 旋转规则（split-half 配对）：
//   i0 = pair, i1 = pair + rotary_dim / 2
//   x'[i0] = x[i0] * cos - x[i1] * sin
//   x'[i1] = x[i1] * cos + x[i0] * sin
//
// q 与 k 必须同 dtype（BF16 或 FP32）；positions 越界（< 0 或 >=
// rope_cache.shape(0)）的元素不写入（kernel 内守卫，host 无法检查 device 数据）。
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方检查。
// 第一阶段：scalar 正确版；bf162 fast path 留后续优化。
// -----------------------------------------------------------------------------
void rope(Tensor* q, Tensor* k, const Tensor& positions, const Tensor& rope_cache,
          unsigned int rotary_dim, const ExecutionContext& ctx);

}  // namespace ccop
