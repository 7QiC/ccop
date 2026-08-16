#pragma once

#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// Embedding 查表（HF F.embedding 的 gather 语义，LLM 前向入口）：
//
//   table:     [vocab_size, d_model]（词嵌入矩阵，只读）
//   token_ids: [num_tokens] int32（device），每个 token 的词表 id
//   out:       [num_tokens, d_model]（独立输出）
//
// 数学形式（按行搬运，无算术）：
//   out[t][d] = table[token_ids[t]][d]
//
// 前置条件：token_ids 的所有值必须在 [0, vocab_size) 内（调用方保证；
// 越界行为未定义，不守卫——与 vLLM/HF 的 gather 语义一致）。
// table 与 out 必须同 dtype（BF16 或 FP32）、各自连续。
//
// 算子只负责计算：参数校验用 assert 兜底，kernel 错误由调用方检查。
// 第一阶段：scalar 正确版，不做性能优化。
// -----------------------------------------------------------------------------
void embed(const Tensor& table, const Tensor& token_ids, Tensor* out, const ExecutionContext& ctx);

}  // namespace ccop
