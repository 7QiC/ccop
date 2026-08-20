#pragma once

namespace ccop {

// Execution context passed to ops by the caller.
//
// stream is a backend-specific opaque handle (e.g. cudaStream_t stored as
// void*); null means the default stream. blas_handle is an optional opaque
// backend BLAS handle (e.g. cublasHandle_t stored as void*), used by gemm.
//
// Context never owns either handle and never allocates memory; buffers are
// owned by the framework/caller.
struct ExecutionContext {
    void* stream = nullptr;
    void* blas_handle = nullptr;
};

}  // namespace ccop
