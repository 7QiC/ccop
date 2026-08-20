#pragma once

#include <cublas_v2.h>

#include <cuda_runtime.h>

#include "ccop/error.h"

namespace ccop {

inline ErrorCode map_cuda_error(cudaError_t error) noexcept {
    switch (error) {
        case cudaSuccess:
            return ErrorCode::kOk;
        case cudaErrorInvalidValue:
            return ErrorCode::kInvalidArgument;
        case cudaErrorMemoryAllocation:
            return ErrorCode::kOutOfMemory;
        default:
            return ErrorCode::kRuntimeError;
    }
}

inline Result<void> check_cuda_launch() noexcept {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return std::unexpected(map_cuda_error(error));
    }
    return {};
}

inline ErrorCode map_cublas_error(cublasStatus_t status) noexcept {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return ErrorCode::kOk;
        case CUBLAS_STATUS_INVALID_VALUE:
            return ErrorCode::kInvalidArgument;
        case CUBLAS_STATUS_ALLOC_FAILED:
            return ErrorCode::kOutOfMemory;
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return ErrorCode::kUnsupported;
        default:
            return ErrorCode::kRuntimeError;
    }
}

inline Result<void> check_cublas(cublasStatus_t status) noexcept {
    if (status != CUBLAS_STATUS_SUCCESS) {
        return std::unexpected(map_cublas_error(status));
    }
    return {};
}

}  // namespace ccop
