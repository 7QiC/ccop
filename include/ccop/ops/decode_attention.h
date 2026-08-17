#pragma once

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Decode 阶段 paged attention（vLLM paged_attention 的 decode 形态，无
// causal mask——当前 token 可见全部历史 KV）：
//
//   q:            [batch_size, num_q_heads, head_dim]（每请求当前一个 token）
//   k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]（paged 池）
//   block_table:  [batch_size, max_blocks_per_req] int32（device），
//                 block_table[b][i] = 请求 b 第 i 个逻辑块的物理块号
//   context_lens: [batch_size] int32（device），请求 b 的 KV token 数
//   out:          [batch_size, num_q_heads, head_dim]（独立输出）
//   scale:        score 缩放系数（调用方传 1/sqrt(head_dim)）
//
// 数学形式（对每个请求 b、每个 q 头 qh，GQA）：
//   kv_head = qh * num_kv_heads / num_q_heads
//   请求 b 的 KV 序列 token s ∈ [0, context_lens[b])：
//     逻辑块 i = s / block_size，块内偏移 r = s % block_size，
//     物理块 blk = block_table[b][i]
//     score[s] = scale * sum_d q[b][qh][d] * k_cache[blk][r][kv_head][d]
//   m = max_s score[s]；l = sum_s exp(score[s] - m)
//   out[b][qh][d] = sum_s (exp(score[s] - m) / l) * v_cache[blk][r][kv_head][d]
//
// q/k_cache/v_cache/out 必须同 dtype（BF16 或 FP32）、各自连续；
// block_table/context_lens 为 int32。前置条件（调用方保证，kernel 不
// 守卫）：block_table 前 ceil(context_lens[b] / block_size) 个块有效且
// 块号在 [0, num_blocks) 内（其余对齐行内容任意）；context_lens 非负。
// num_q_heads 须为 num_kv_heads 的整数倍。
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方检查。
// 第一阶段：scalar 正确版（每线程一个 (b, qh) 输出行），不做性能优化。
// -----------------------------------------------------------------------------
void decode_attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                      const Tensor& block_table, const Tensor& context_lens, Tensor* out,
                      float scale, const ExecutionContext& ctx);

}  // namespace ccop
