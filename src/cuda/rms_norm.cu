#include "ccop/ops/rms_norm.h"

#include <cassert>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccop {
namespace {

// TODO(operator): implement the rms_norm kernel.
//
// CUDA 惯例：__global__ 模板函数，返回 void。
// 模板参数：ActivationT（输入/输出 dtype）、WeightT（权重 dtype）。
//
// 第一阶段（正确性为先，不做性能优化）：
//   - naive 正确版本：每个线程处理一行；
//   - 先对行内 dim 个元素累加平方和（用 float 累加，避免低精度溢出）：
//       float sum = 0; for j: sum += (float)input[i*dim+j]^2;
//       float rms = sqrt(sum / dim + eps);
//   - 再归一化并乘权重写回：
//       output[i*dim+j] = (ActivationT)((float)input[i*dim+j] / rms * (float)weight[j]);
//   - 越界线程直接 return（grid 按 rows 向上取整）。
template <typename ActivationT, typename WeightT>
__global__ void rms_norm_kernel(const ActivationT* input, const WeightT* weight,
                                ActivationT* output, unsigned int rows, unsigned int dim,
                                float eps) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= rows) {
        return;
    }
    float sum = 0;
    for (unsigned int i = 0; i < dim; ++i) {
        float v = static_cast<float>(input[tid * dim + i]);
        sum += v * v;
    }
    float rms = sqrt(sum / dim + eps);
    for (unsigned int i = 0; i < dim; ++i) {
        output[tid * dim + i] = (ActivationT)((float)input[tid * dim + i] / rms * (float)weight[i]);
    }
}

}  // namespace

void rms_norm(Tensor* out, const Tensor& input, const Tensor& weight, float eps,
              const ExecutionContext& ctx) {
    assert(out != nullptr && out->valid());
    assert(input.valid() && weight.valid());
    assert(input.rank() == 2);
    assert(input.is_contiguous());
    assert(weight.rank() == 1);
    assert(weight.is_contiguous());
    assert(input.shape(1) == weight.shape(0));
    assert(eps > 0.0f);

    const unsigned int rows = static_cast<unsigned int>(input.shape(0));
    const unsigned int dim = static_cast<unsigned int>(input.shape(1));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    const unsigned int block = 256;
    const unsigned int grid = static_cast<unsigned int>((rows + block - 1) / block);

    // 组合 dtype 分发：activation dtype × weight dtype → kernel 模板实例。
    // 先用 BF16×FP32（LLM 标准）和 FP32×FP32 两种组合，后续按需扩展。
    // TODO(operator): 在下面每个 case 中补 launch：
    //   1. 解包裸指针并转成对应模板类型；
    //   2. stream：cudaStream_t s = static_cast<cudaStream_t>(ctx.stream)；
    //   3. naive：每个线程一行，block = 256，grid = (rows + block - 1) / block；
    //   4. kernel<<<grid, block, 0, s>>>(in, w, out_ptr, rows, dim, eps)。
    switch (input.dtype()) {
        case DType::kBFloat16: {
            switch (weight.dtype()) {
                case DType::kFloat32:
                    rms_norm_kernel<__nv_bfloat16, float><<<grid, block, 0, s>>>(
                        static_cast<const __nv_bfloat16*>(input.data()),
                        static_cast<const float*>(weight.data()),
                        static_cast<__nv_bfloat16*>(out->data()), rows, dim, eps);
                    break;
                default:
                    return;  // 不支持的权重 dtype 组合
            }
            break;
        }
        case DType::kFloat32: {
            switch (weight.dtype()) {
                case DType::kFloat32:
                    rms_norm_kernel<float, float><<<grid, block, 0, s>>>(
                        static_cast<const float*>(input.data()),
                        static_cast<const float*>(weight.data()),
                        static_cast<float*>(out->data()), rows, dim, eps);
                    break;
                default:
                    return;
            }
            break;
        }
        default:
            return;  // 不支持的 activation dtype
    }
}

}  // namespace ccop
