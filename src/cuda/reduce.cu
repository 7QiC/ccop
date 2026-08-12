#include "ccop/ops/reduce.h"

#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

namespace ccop {
namespace {

// TODO(operator): implement the row-sum kernel.
//
// CUDA 惯例：__global__ 模板函数，返回 void。
// 第一阶段（正确性为先，不做性能优化）：
//   - host 入口已解析好标量参数（rows/cols）；
//   - naive 正确版本：每个线程处理一行，累加该行 cols 个元素：
//       output[i] = sum_j input[i * cols + j]
//   - 越界线程直接 return（grid 按 rows 向上取整）。
template <typename T>
__global__ void reduce_sum_rows_kernel(const T* input, T* output, unsigned int rows,
                                       unsigned int cols) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= rows) {
        return;
    }
    T sum = 0;
    for (unsigned int i = 0; i < cols; ++i) {
        sum +=  input[tid * cols + i];
    }
    output[tid] = sum;
}

}  // namespace

void reduce_sum_rows(Tensor* out, const Tensor& input, const ExecutionContext& ctx) {
    assert(out != nullptr && out->valid());
    assert(input.valid());
    assert(input.rank() == 2);
    assert(input.is_contiguous());

    const unsigned int rows = static_cast<unsigned int>(input.shape(0));
    const unsigned int cols = static_cast<unsigned int>(input.shape(1));
    assert(out->rank() == 1);
    assert(out->shape(0) == rows);

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    const unsigned int block = 256;
    const unsigned int grid = static_cast<unsigned int>((rows + block - 1) / block);

    // TODO(operator): launch reduce_sum_rows_kernel。
    // 提示：
    //   1. 解包裸指针：const T* in = static_cast<const T*>(input.data());
    //      T* out_ptr = static_cast<T*>(out->data());
    //   2. stream：cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    //   3. 每个线程一行：block = 256，grid = (rows + block - 1) / block；
    //   4. 用非默认流 launch：kernel<<<grid, block, 0, s>>>(in, out_ptr, rows, cols)。
    // kernel 错误由调用方检查（cudaGetLastError/同步），这里不返回。
    switch (input.dtype()) {
        case DType::kFloat32:
            reduce_sum_rows_kernel<float><<<grid, block, 0, s>>>(
                static_cast<const float*>(input.data()), static_cast<float*>(out->data()), rows,
                cols);
            break;
        default:
            return;
    }
}

}  // namespace ccop
