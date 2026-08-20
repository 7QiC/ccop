#include "ccop/ops/gemm.h"

#include <cstdint>
#include <cublas_v2.h>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ccop/cuda/error_cuda.h"

namespace ccop {
namespace {

bool to_dim(std::int64_t value, unsigned int& out) noexcept {
    if (value <= 0 || static_cast<std::uint64_t>(value) >
                          static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
        return false;
    }
    out = static_cast<unsigned int>(value);
    return true;
}

cublasOperation_t cublas_op(bool trans) noexcept { return trans ? CUBLAS_OP_T : CUBLAS_OP_N; }

}  // namespace

Result<void> gemm(Tensor* c, const Tensor& a, const Tensor& b, bool trans_a, bool trans_b,
                  float alpha, float beta, const ExecutionContext& ctx) {
    if (c == nullptr || !c->valid() || !a.valid() || !b.valid()) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!c->is_cuda() || !a.is_cuda() || !b.is_cuda()) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (a.rank() != 2 || b.rank() != 2 || c->rank() != 2) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (a.stride(1) != 1 || b.stride(1) != 1 || c->stride(1) != 1) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }

    unsigned int m = 0;
    unsigned int n = 0;
    unsigned int k = 0;
    unsigned int k_b = 0;
    if (!to_dim(trans_a ? a.shape(1) : a.shape(0), m) ||
        !to_dim(trans_b ? b.shape(0) : b.shape(1), n) ||
        !to_dim(trans_a ? a.shape(0) : a.shape(1), k) ||
        !to_dim(trans_b ? b.shape(1) : b.shape(0), k_b)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (k != k_b) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (c->shape(0) != static_cast<std::int64_t>(m) ||
        c->shape(1) != static_cast<std::int64_t>(n)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const std::int64_t lda = a.stride(0);
    const std::int64_t ldb = b.stride(0);
    const std::int64_t ldc = c->stride(0);
    if (lda < a.shape(1) || ldb < b.shape(1) || ldc < c->shape(1)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }

    cudaDataType_t in_type = CUDA_R_32F;
    cudaDataType_t out_type = CUDA_R_32F;
    switch (a.dtype()) {
        case DType::kBFloat16:
            if (b.dtype() != DType::kBFloat16) {
                return std::unexpected(ErrorCode::kInvalidArgument);
            }
            in_type = CUDA_R_16BF;
            if (c->dtype() == DType::kBFloat16) {
                out_type = CUDA_R_16BF;
            } else if (c->dtype() == DType::kFloat32) {
                out_type = CUDA_R_32F;
            } else {
                return std::unexpected(ErrorCode::kUnsupported);
            }
            break;
        case DType::kFloat32:
            if (b.dtype() != DType::kFloat32 || c->dtype() != DType::kFloat32) {
                return std::unexpected(ErrorCode::kInvalidArgument);
            }
            break;
        default:
            return std::unexpected(ErrorCode::kUnsupported);
    }

    if (ctx.blas_handle == nullptr) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    auto* handle = static_cast<cublasHandle_t>(ctx.blas_handle);
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream);
    if (stream != nullptr) {
        if (auto r = check_cublas(cublasSetStream(handle, stream)); !r) {
            return r;
        }
    }

    const cublasOperation_t op_a = cublas_op(trans_a);
    const cublasOperation_t op_b = cublas_op(trans_b);

    // Row-major C = op(A) @ op(B) maps to column-major
    // C^T = op(B)^T @ op(A)^T, which is cuBLAS's B-first argument order.
    return check_cublas(cublasGemmEx(handle, op_b, op_a, n, m, k, &alpha, b.data(), in_type, ldb,
                                     a.data(), in_type, lda, &beta, c->data(), out_type, ldc,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace ccop
