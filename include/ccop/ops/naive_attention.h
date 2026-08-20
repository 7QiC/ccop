#pragma once

#include "ccop/error.h"
#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Naive attention（HF eager attention 的朴素形态，causal mask）：
//
//   q:   [num_tokens, num_q_heads, head_dim]
//   k/v: [num_tokens, num_kv_heads, head_dim]（完整 KV，非 paged）
//   out: [num_tokens, num_q_heads, head_dim]（独立输出）
//   scale: score 的缩放系数（调用方传 1/sqrt(head_dim)）
//
// 数学形式（对每个 token t、每个 q 头 qh，GQA）：
//   kv_head = qh * num_kv_heads / num_q_heads   （要求 num_q_heads 为
//             num_kv_heads 的整数倍，组内 q 头共享同一 kv 头）
//   score[s] = scale * sum_d q[t][qh][d] * k[s][kv_head][d]   （s = 0..t）
//   m = max_s score[s]；l = sum_s exp(score[s] - m)
//   p[s] = exp(score[s] - m) / l
//   out[t][qh][d] = sum_s p[s] * v[s][kv_head][d]
//
// 四个数组必须同 dtype（BF16 或 FP32）、各自连续。token t 只可见
// [0, t] 的 KV（causal mask；paged 形态留给 prefill/decode 算子）。
//
// 算子只负责计算：参数错误返回 ErrorCode；kernel 错误映射后返回 ErrorCode。
// 第一阶段：scalar 正确版（每线程一个输出行），不做性能优化。
// -----------------------------------------------------------------------------
Result<void> naive_attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor* out,
                             float scale, const ExecutionContext& ctx);

}  // namespace ccop
