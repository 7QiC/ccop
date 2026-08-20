#pragma once

#include "ccop/error.h"
#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// KV cache 写入（paged attention 的 KV 落盘，对应 vLLM reshape_and_cache
// 后半段）：
//
//   k_new/v_new: [num_tokens, num_kv_heads, head_dim]（本轮新算出的 K/V）
//   k_cache/v_cache: [num_slots, num_kv_heads, head_dim]（paged cache 池，被写入）
//   slot_mapping: [num_tokens] int32（device），每个 token 的目标 slot 号
//
// 数学形式（按 slot 散写，纯搬运无算术）：
//   对每个 token t，slot = slot_mapping[t]：
//     若 0 <= slot < num_slots：
//       k_cache[slot][h][d] = k_new[t][h][d]
//       v_cache[slot][h][d] = v_new[t][h][d]
//     否则（slot < 0 或 >= num_slots，padding/无效 token）跳过，不写任何位置。
//
// 四个 KV 数组必须同 dtype（BF16 或 FP32）、各自连续；slot 越界检查在
// kernel 内做（slot_mapping 在 device，host 无法校验）。
//
// 算子只负责计算：参数错误返回 ErrorCode；kernel 错误映射后返回 ErrorCode。
// 第一阶段：scalar 正确版，不做性能优化。
// -----------------------------------------------------------------------------
Result<void> write_kv_cache(const Tensor& k_new, const Tensor& v_new, Tensor* k_cache,
                            Tensor* v_cache, const Tensor& slot_mapping,
                            const ExecutionContext& ctx);

}  // namespace ccop
