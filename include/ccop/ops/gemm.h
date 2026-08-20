#pragma once

#include "ccop/error.h"
#include "ccop/execution_context.h"
#include "ccop/tensor.h"

namespace ccop {

// -----------------------------------------------------------------------------
// 通用稠密 GEMM（当前 backend 封装 cuBLAS，后续可替换为自研 kernel）：
//
//   C = alpha * op(A) @ op(B) + beta * C
//
// 输入均为 2D 行主序视图：
//   A: [M, K]（trans_a = false）或 [K, M]（trans_a = true）
//   B: [K, N]（trans_b = false）或 [N, K]（trans_b = true）
//   C: [M, N]
//
// 视图要求 stride(1) == 1；leading dimension 取 stride(0)（因此支持
// 带 padding 的连续行）。A/B 必须同 dtype，支持：
//   BF16 × BF16 → BF16（FP32 累加）
//   BF16 × BF16 → FP32（logits GEMM）
//   FP32 × FP32 → FP32
//
// 当前 CUDA backend 要求 ctx.blas_handle 为有效 cublasHandle_t；
// ctx.stream 为 cudaStream_t，null 表示默认流。
// 算子只负责计算：参数错误返回 ErrorCode；backend 错误映射后返回 ErrorCode。
// -----------------------------------------------------------------------------
Result<void> gemm(Tensor* c, const Tensor& a, const Tensor& b, bool trans_a, bool trans_b,
                  float alpha, float beta, const ExecutionContext& ctx);

}  // namespace ccop
