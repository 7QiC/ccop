#include "ccop/ops/silu_mul.h"

#include <cassert>
#include <cmath>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccop {
namespace {

// SiluMul kernel 声明（接口已定，函数体待实现）。
// 模板参数：T（gate/up/out 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版，不做性能优化）。
//
// 数学形式（逐元素）：
//   out[i] = gate[i] / (1 + exp(-gate[i])) * up[i]
//
// 数据布局（全部行主序连续一维）：
//   gate/up/out: [n]，三个独立数组，out[i] 只依赖 gate[i]/up[i]。
//
// 并行映射：
//   grid.x 覆盖全部 n 个元素（block 内线程数向上取整，需越界守卫）；
//   blockIdx.x/threadIdx.x → 元素索引 i。
//
// 数值注意：BF16 参与计算前提升到 float，写回前转回原 dtype；
// sigmoid 在负 x 大绝对值时 exp(-x) 会溢出，注意计算形式（float 下
// 1/(1+exp(-x)) 对 x 为负是安全的，正值饱和到 1 也符合 sigmoid 语义）。
template <typename T>
__global__ void silu_mul_kernel(const T* gate, const T* up, T* out, unsigned int n) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) {
        return;
    }
    out[tid] =
        static_cast<T>(static_cast<float>(gate[tid]) / (1 + expf(static_cast<float>(-gate[tid]))) *
                       static_cast<float>(up[tid]));
}

}  // namespace

void silu_mul(Tensor* out, const Tensor& gate, const Tensor& up, const ExecutionContext& ctx) {
    assert(out != nullptr && out->valid());
    assert(gate.valid() && up.valid());
    assert(gate.rank() == 1 && gate.is_contiguous());
    assert(up.rank() == 1 && up.is_contiguous());
    assert(out->rank() == 1 && out->is_contiguous());
    assert(gate.dtype() == up.dtype());
    assert(gate.dtype() == out->dtype());
    assert(gate.shape(0) == up.shape(0));
    assert(out->shape(0) == gate.shape(0));

    const unsigned int n = static_cast<unsigned int>(gate.shape(0));

    const unsigned int block = 256;
    const unsigned int grid = (n + block - 1) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);

    // 组合 dtype 分发：gate/up/out 同 dtype → kernel 模板实例。
    // TODO(operator): 在下面每个 case 中补 launch。
    //   并行映射（与 kernel 注释一致）：
    //     block = 256（scalar 版每线程一个元素）；
    //     grid = ceil(n / block)。
    //   调用要点：
    //     stream 从 ctx.stream 取；gate/up/out 的裸指针从各自 Tensor::data()
    //     解包并转成 kernel 参数类型；模板参数 T 按 dtype 取
    //     __nv_bfloat16（kBFloat16）或 float（kFloat32）；标量按签名顺序传。
    switch (gate.dtype()) {
        case DType::kBFloat16:
            silu_mul_kernel<__nv_bfloat16>
                <<<grid, block, 0, s>>>(static_cast<const __nv_bfloat16*>(gate.data()),
                                        static_cast<const __nv_bfloat16*>(up.data()),
                                        static_cast<__nv_bfloat16*>(out->data()), n);
            break;
        case DType::kFloat32:
            silu_mul_kernel<float>
                <<<grid, block, 0, s>>>(static_cast<const float*>(gate.data()),
                                        static_cast<const float*>(up.data()),
                                        static_cast<float*>(out->data()), n);
            break;
        default:
            return;  // 不支持的 dtype
    }
}

}  // namespace ccop
