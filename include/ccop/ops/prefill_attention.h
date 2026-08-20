#pragma once

#include "ccop/error.h"
#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Prefill 阶段 paged attention（vLLM paged_attention 的 prefill 形态，带
// causal mask：每个 prompt token 只可见自己及之前的 KV）：
//
//   q:            [total_q_tokens, num_q_heads, head_dim]（各请求 prompt 拼接）
//   k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]（paged 池）
//   block_table:  [batch_size, max_blocks_per_req] int32（device），
//                 block_table[b][i] = 请求 b 第 i 个逻辑块的物理块号
//   query_start_loc: [batch_size + 1] int32（device，单调递增），
//                 query_start_loc[0] = 0，请求 b 的 prompt token 占据 q 的
//                 [query_start_loc[b], query_start_loc[b+1]) 行
//   context_lens: [batch_size] int32（device），请求 b 的总可见 KV 长度
//                 （= 前缀 token 数 + 本次 prompt 长度；chunked prefill /
//                 prefix cache 场景下前缀已在 cache 中）
//   out:          [total_q_tokens, num_q_heads, head_dim]（独立输出）
//   scale:        score 缩放系数（调用方传 1/sqrt(head_dim)）
//
// 数学形式（对每个请求 b、其每个 prompt token tq、每个 q 头 qh，GQA）：
//   kv_head = qh * num_kv_heads / num_q_heads
//   prompt_len = query_start_loc[b+1] - query_start_loc[b]
//   prefix_len = context_lens[b] - prompt_len
//   p = tq - query_start_loc[b]（该 token 在本次 prompt 内的偏移）
//   causal：可见 KV token s ∈ [0, prefix_len + p + 1)（前缀 + 自己及之前）
//     逻辑块 i = s / block_size，块内偏移 r = s % block_size，
//     物理块 blk = block_table[b][i]
//     score[s] = scale * sum_d q[tq][qh][d] * k_cache[blk][r][kv_head][d]
//   m = max_s score[s]；l = sum_s exp(score[s] - m)
//   out[tq][qh][d] = sum_s (exp(score[s] - m) / l) * v_cache[blk][r][kv_head][d]
//
// q/k_cache/v_cache/out 必须同 dtype（BF16 或 FP32）、各自连续；
// block_table/query_start_loc/context_lens 为 int32。前置条件（调用方
// 保证，kernel 不守卫）：block_table 前 ceil(context_lens[b] /
// block_size) 个块有效且块号在 [0, num_blocks) 内；query_start_loc
// 单调递增且首尾与 total_q_tokens 一致；context_lens[b] >= prompt_len[b]。
// num_q_heads 须为 num_kv_heads 的整数倍。
//
// 算子只负责计算：参数错误返回 ErrorCode；kernel 错误映射后返回 ErrorCode。
// 第一阶段：scalar 正确版（每线程一个 (tq, qh) 输出行），不做性能优化。
// -----------------------------------------------------------------------------
Result<void> prefill_attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                               const Tensor& block_table, const Tensor& query_start_loc,
                               const Tensor& context_lens, Tensor* out, float scale,
                               const ExecutionContext& ctx);

}  // namespace ccop
