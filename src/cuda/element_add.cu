
#include "ccop/ops/element_add.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ccop/cuda/error_cuda.h"

namespace ccop {
namespace {

// ElementAdd kernel 声明（接口已定，函数体待实现）。
// 模板参数：T（dst/src 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版，不做性能优化）。
//
// 数学形式（逐元素原地加）：
//   dst[i] ← dst[i] + src[i]
//
// 数据布局（行主序连续一维）：
//   dst/src: [n]，两个独立数组，元素 i 线性偏移即 i。
//
// 并行映射：
//   grid.x 覆盖全部 n 个元素（block 内线程数向上取整，需越界守卫）；
//   blockIdx.x/threadIdx.x → 元素索引 i。
//
// 数值注意：
//   - BF16 参与计算前提升到 float，写回前转回原 dtype；
//   - 原地语义：每个元素只读写自身，线程间无依赖，无需同步。
template <typename T>
__global__ void element_add_kernel(T* dst, const T* src, unsigned int n) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) {
        return;
    }
    dst[tid] = static_cast<T>(static_cast<float>(dst[tid]) + static_cast<float>(src[tid]));
}

}  // namespace

Result<void> element_add(Tensor* dst, const Tensor& src, const ExecutionContext& ctx) {
    if (!(dst != nullptr && dst->valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(src.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(dst->rank() == 1 && dst->is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(src.rank() == 1 && src.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(dst->dtype() == src.dtype())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(dst->shape(0) == src.shape(0))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const unsigned int n = static_cast<unsigned int>(dst->shape(0));
    if (!(n > 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const unsigned int block = 256;
    const unsigned int grid = (n + block - 1) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    // 组合 dtype 分发：dst/src 同 dtype → kernel 模板实例。
    //   并行映射（与 kernel 注释一致）：
    //     block = 256（scalar 版每线程一个元素）；
    //     grid = ceil(n / block)。
    //   调用要点：
    //     stream 从 ctx.stream 取；dst/src 的裸指针从各自 Tensor::data()
    //     解包并转成 kernel 参数类型；模板参数 T 按 dtype 取
    //     __nv_bfloat16（kBFloat16）或 float（kFloat32）；标量按签名顺序传。
    switch (dst->dtype()) {
        case DType::kBFloat16:
            element_add_kernel<__nv_bfloat16>
                <<<grid, block, 0, s>>>(static_cast<__nv_bfloat16*>(dst->data()),
                                        static_cast<const __nv_bfloat16*>(src.data()), n);
            return check_cuda_launch();
        case DType::kFloat32:
            element_add_kernel<float><<<grid, block, 0, s>>>(
                static_cast<float*>(dst->data()), static_cast<const float*>(src.data()), n);
            return check_cuda_launch();
        default:
            return std::unexpected(ErrorCode::kUnsupported);
    }
}

}  // namespace ccop
