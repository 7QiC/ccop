#pragma once

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Greedy 采样（生成闭环出口，HF argmax 语义）：
//
//   logits: [num_tokens, vocab_size] float32（模型最终输出）
//   tokens: [num_tokens] int32（独立输出，下一个 token id）
//
// 数学形式（逐行 argmax）：
//   tokens[t] = argmax_v logits[t][v]
//   平局（多个 v 同为最大值）时取最小 v（严格 > 才更新）。
//
// logits 必须 float32、连续；logits 含 NaN 时行为未定义（调用方保证）。
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方检查。
// 第一阶段：scalar 正确版（每线程一行），不做性能优化。
// -----------------------------------------------------------------------------
void greedy_sample(const Tensor& logits, Tensor* tokens, const ExecutionContext& ctx);

}  // namespace ccop
