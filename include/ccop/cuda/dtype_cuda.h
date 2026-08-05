#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ccop/dtype.h"

namespace ccop {

template <>
struct NativeOf<Float16Tag> {
    using type = __half;
};

template <>
struct NativeOf<BFloat16Tag> {
    using type = __nv_bfloat16;
};

}  // namespace ccop

static_assert(sizeof(__half) == ccop::dtype_size_v<ccop::Float16Tag>);
static_assert(sizeof(__nv_bfloat16) == ccop::dtype_size_v<ccop::BFloat16Tag>);
