#include "ccop/ops/softmax.h"

#include <cassert>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccop {
namespace {

// Softmax kernel 声明（接口已定，函数体待实现）。
// 模板参数：T（input/out 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版，不做性能优化）。
//
// 数学形式（逐行，数值稳定）：
//   m_i = max_j x[i][j]
//   out[i][j] = exp(x[i][j] - m_i) / sum_j exp(x[i][j] - m_i)
//
// 数据布局（行主序连续二维）：
//   input/out: [rows, cols]，x[i][j] 线性偏移 = i * cols + j。
//
// 并行映射：
//   grid.x 覆盖 rows（block 内线程数向上取整，需越界守卫）；
//   blockIdx.x/threadIdx.x → 行索引 i，每线程完整处理一行（cols 个元素）。
//
// 数值注意：
//   - 先求行内 max 再减，避免 exp 溢出（输入可能含大正数）；
//   - BF16 参与计算前提升到 float，写回前转回原 dtype；
//   - 行内累加（sum）用 float 精度。
template <typename T>
__global__ void softmax_kernel(const T* input, T* out, unsigned int rows, unsigned int cols) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= rows) {
        return;
    }
    T max = input[cols * tid];
    for (unsigned int i = 0; i < cols; ++i) {
        T tmp = input[cols * tid + i];
        max = max > tmp ? max : tmp;
    }
    float sum = 0;
    for (unsigned int i = 0; i < cols; ++i) {
        sum += expf(static_cast<float>(input[cols * tid + i]) - static_cast<float>(max));
    }
    for (unsigned int i = 0; i < cols; ++i) {
        out[cols * tid + i] = static_cast<T>(
            expf(static_cast<float>(input[cols * tid + i]) - static_cast<float>(max)) / sum);
    }
}

}  // namespace

void softmax(Tensor* out, const Tensor& input, const ExecutionContext& ctx) {
    assert(out != nullptr && out->valid());
    assert(input.valid());
    assert(input.rank() == 2 && input.is_contiguous());
    assert(out->rank() == 2 && out->is_contiguous());
    assert(out->dtype() == input.dtype());
    assert(out->shape(0) == input.shape(0));
    assert(out->shape(1) == input.shape(1));

    const unsigned int rows = static_cast<unsigned int>(input.shape(0));
    const unsigned int cols = static_cast<unsigned int>(input.shape(1));
    assert(rows > 0 && cols > 0);

    const unsigned int block = 256;
    const unsigned int grid = (rows + block - 1) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);

    // 组合 dtype 分发：input/out 同 dtype → kernel 模板实例。
    // TODO(operator): 在下面每个 case 中补 launch。
    //   并行映射（与 kernel 注释一致）：
    //     block = 256（scalar 版每线程一行）；
    //     grid = ceil(rows / block)。
    //   调用要点：
    //     stream 从 ctx.stream 取；input/out 的裸指针从各自 Tensor::data()
    //     解包并转成 kernel 参数类型；模板参数 T 按 dtype 取
    //     __nv_bfloat16（kBFloat16）或 float（kFloat32）；标量按签名顺序传。
    switch (input.dtype()) {
        case DType::kBFloat16:
            softmax_kernel<__nv_bfloat16>
                <<<grid, block, 0, s>>>(static_cast<const __nv_bfloat16*>(input.data()),
                                        static_cast<__nv_bfloat16*>(out->data()), rows, cols);
            break;
        case DType::kFloat32:
            softmax_kernel<float><<<grid, block, 0, s>>>(static_cast<const float*>(input.data()),
                                                         static_cast<float*>(out->data()), rows,
                                                         cols);
            break;
        default:
            return;  // 不支持的 dtype
    }
}

}  // namespace ccop
