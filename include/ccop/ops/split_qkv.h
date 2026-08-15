#pragma once

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// QKV 拆分（Qwen3/LLaMA attention 的 qkv_proj 输出重排）：
//
//   qkv: [num_tokens, qkv_dim]，qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
//   q:   [num_tokens, num_q_heads, head_dim]（独立输出）
//   k:   [num_tokens, num_kv_heads, head_dim]（独立输出）
//   v:   [num_tokens, num_kv_heads, head_dim]（独立输出）
//
// 每个 token 的行内布局（全部行主序平铺）：
//   qkv[t] = [ Q | K | V ]
//     Q 段：num_q_heads * head_dim 个元素，q[t][h][d] = qkv[t][h * head_dim + d]
//     K 段：num_kv_heads * head_dim 个元素，k[t][h][d] = qkv[t][q_dim + h * head_dim + d]
//     V 段：num_kv_heads * head_dim 个元素，v[t][h][d] = qkv[t][q_dim + k_dim + h * head_dim + d]
//   其中 q_dim = num_q_heads * head_dim，k_dim = v_dim = num_kv_heads * head_dim。
//
// 四个数组必须同 dtype（BF16 或 FP32）、各自连续；q/k/v 是独立输出
// （可与 qkv 是同一 buffer 的不同切片，只要求各自视图连续）。
// 纯数据搬运，无算术。
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方检查。
// 第一阶段：scalar 正确版，不做性能优化。
// -----------------------------------------------------------------------------
void split_qkv(const Tensor& qkv, Tensor* q, Tensor* k, Tensor* v, unsigned int num_q_heads,
               unsigned int num_kv_heads, const ExecutionContext& ctx);

}  // namespace ccop
