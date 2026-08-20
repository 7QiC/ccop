#pragma once

#include "ccop/error.h"
#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Greedy 采样（生成闭环出口，HF argmax 语义）：
//
//   logits:         [num_tokens, vocab_size] float32（模型最终输出）
//   logits_indices: [batch_size] int32（device），
//                   logits_indices[b] = 请求 b 要采样的 logits 行号
//   tokens:         [batch_size] int32（独立输出，下一个 token id）
//
// 数学形式（按索引逐行 argmax）：
//   row = logits_indices[b]
//   若 0 <= row < num_tokens：
//     tokens[b] = argmax_v logits[row][v]
//   否则跳过（padding/无效请求），tokens[b] 保持原值。
//   平局（多个 v 同为最大值）时取最小 v（严格 > 才更新）。
//
// logits/logits_indices 必须各自连续；logits 固定 float32，
// logits_indices/tokens 为 int32。logits 含 NaN 时行为未定义（调用方保证）。
//
// 算子只负责计算：参数错误返回 ErrorCode；kernel 错误映射后返回 ErrorCode。
// 第一阶段：scalar 正确版（每线程一行），不做性能优化。
// -----------------------------------------------------------------------------
Result<void> greedy_sample(const Tensor& logits, const Tensor& logits_indices, Tensor* tokens,
                           const ExecutionContext& ctx);

}  // namespace ccop
