#include "ccop/ops/embed.h"

#include <cassert>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccop {
namespace {

// Embed kernel 声明（接口已定，函数体待实现）。
// 模板参数：T（table/out 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版，不做性能优化）。
//
// 数学形式（按行搬运，无算术）：
//   out[t][d] = table[token_ids[t]][d]
//
// 数据布局（全部行主序连续）：
//   table: [vocab_size, d_model]，线性偏移 = v * d_model + d；
//   token_ids: [num_tokens] int32；
//   out: [num_tokens, d_model]，线性偏移 = t * d_model + d。
//
// 并行映射（输出驱动）：
//   grid.x 与 block 内 threadIdx.x 一起平铺 out 的全部
//   num_tokens * d_model 个输出元素（block 向上取整，需越界守卫）；
//   blockIdx.x/threadIdx.x → 输出元素线性索引 idx →
//   t = idx / d_model、d = idx % d_model。
//
// 数值注意：
//   - 纯搬运：同 dtype 直接赋值，无需 float 提升转换；
//   - token_ids 合法性（[0, vocab_size)）由调用方保证，kernel 不守卫；
//   - 维度与索引统一 unsigned int（uint32 上限约 42.9 亿，覆盖
//     32 头 × 100 万上下文 × head_dim 的目标场景）。
template <typename T>
__global__ void embed_kernel(const T* table, const int32_t* token_ids, T* out,
                             unsigned int num_tokens, unsigned int d_model) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_tokens * d_model) {
        return;
    }
    unsigned int t = tid / d_model;
    unsigned int d = tid % d_model;
    int32_t token_id = token_ids[t];
    out[tid] = table[d_model * token_id + d];
}

}  // namespace

void embed(const Tensor& table, const Tensor& token_ids, Tensor* out, const ExecutionContext& ctx) {
    assert(table.valid());
    assert(table.rank() == 2 && table.is_contiguous());
    assert(out != nullptr && out->valid());
    assert(out->rank() == 2 && out->is_contiguous());
    assert(out->dtype() == table.dtype());

    const unsigned int num_tokens = static_cast<unsigned int>(out->shape(0));
    const unsigned int d_model = static_cast<unsigned int>(table.shape(1));
    assert(num_tokens > 0 && d_model > 0);
    assert(table.shape(0) > 0);  // vocab_size
    assert(out->shape(1) == static_cast<std::int64_t>(d_model));

    assert(token_ids.valid());
    assert(token_ids.rank() == 1 && token_ids.is_contiguous());
    assert(token_ids.dtype() == DType::kInt32);
    assert(token_ids.shape(0) == static_cast<std::int64_t>(num_tokens));

    // 组合 dtype 分发：table/out 同 dtype → kernel 模板实例。
    // TODO(operator): 在下面每个 case 中补 launch。
    //   并行映射（与 kernel 注释一致，输出驱动）：
    //     block = 256（scalar 版每线程一个输出元素）；
    //     grid = ceil(num_tokens * d_model / block)。
    //   调用要点：
    //     stream 从 ctx.stream 取；table/token_ids/out 的裸指针从各自
    //     Tensor::data() 解包并转成 kernel 参数类型（token_ids 为
    //     const int32_t*）；模板参数 T 按 dtype 取 __nv_bfloat16
    //     （kBFloat16）或 float（kFloat32）；标量按签名顺序传
    //     （num_tokens, d_model）。
    const unsigned int block = 256;
    const unsigned int grid = (num_tokens * d_model - 1 + block) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    switch (table.dtype()) {
        case DType::kBFloat16:
            embed_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
                static_cast<const __nv_bfloat16*>(table.data()),
                static_cast<const int32_t*>(token_ids.data()),
                static_cast<__nv_bfloat16*>(out->data()), num_tokens, d_model);
            break;
        case DType::kFloat32:
            embed_kernel<float>
                <<<grid, block, 0, s>>>(static_cast<const float*>(table.data()),
                                        static_cast<const int32_t*>(token_ids.data()),
                                        static_cast<float*>(out->data()), num_tokens, d_model);
            break;
        default:
            return;  // 不支持的 dtype
    }
}

}  // namespace ccop
