#pragma once

namespace ccop {

// Execution context passed to ops by the caller. stream is a backend-specific
// opaque handle (e.g. cudaStream_t stored as void*); null means the default
// stream. Memory ownership never lives here: buffers are owned by the
// framework/caller, so no allocator is part of this context.
struct ExecutionContext {
    void* stream = nullptr;
};

}  // namespace ccop
