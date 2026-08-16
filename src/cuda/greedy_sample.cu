#include "ccop/ops/greedy_sample.h"

#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

namespace ccop {
namespace {

// GreedySample kernel 声明（接口已定，函数体待实现）。
// 无模板：logits 按采样上游约定固定 float32（不需要 dtype 分发）。
// 第一阶段（正确性为先，scalar 版：每线程一行）。
//
// 数学形式（逐行 argmax）：
//   tokens[t] = argmax_v logits[t][v]，平局取最小 v（严格 > 才更新）。
//
// 数据布局（行主序连续）：
//   logits: [num_tokens, vocab_size]，线性偏移 = t * vocab_size + v；
//   tokens: [num_tokens] int32。
//
// 并行映射：
//   grid.x 与 block 内 threadIdx.x 一起平铺 num_tokens 行
//   （block 向上取整，需越界守卫）；
//   blockIdx.x/threadIdx.x → 行索引 t，每线程遍历该行全部 vocab_size 个
//   元素并维护 argmax（初始可设 v = 0）。
//
// 数值注意：
//   - 纯比较无算术：平局语义用严格 > 更新，保证保留最小索引；
//   - logits 含 NaN 时行为未定义（调用方保证）；
//   - 维度与索引统一 unsigned int（uint32 上限约 42.9 亿，覆盖
//     32 头 × 100 万上下文 × head_dim 的目标场景）。
__global__ void greedy_sample_kernel(const float* logits, int32_t* tokens, unsigned int num_tokens,
                                     unsigned int vocab_size) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_tokens) {
        return;
    }
    tokens[tid] = 0;
    float v = logits[tid * vocab_size];
    for (unsigned int i = 1; i < vocab_size; ++i) {
        float tmp = logits[tid * vocab_size + i];
        if (tmp > v) {
            v = tmp;
            tokens[tid] = i;
        }
    }
}

}  // namespace

void greedy_sample(const Tensor& logits, Tensor* tokens, const ExecutionContext& ctx) {
    assert(logits.valid());
    assert(logits.rank() == 2 && logits.is_contiguous());
    assert(logits.dtype() == DType::kFloat32);
    assert(tokens != nullptr && tokens->valid());
    assert(tokens->rank() == 1 && tokens->is_contiguous());
    assert(tokens->dtype() == DType::kInt32);

    const unsigned int num_tokens = static_cast<unsigned int>(logits.shape(0));
    const unsigned int vocab_size = static_cast<unsigned int>(logits.shape(1));
    assert(num_tokens > 0 && vocab_size > 0);
    assert(tokens->shape(0) == static_cast<std::int64_t>(num_tokens));

    // TODO(operator): 补 launch（单 dtype：logits 固定 float32，无分发 switch）。
    //   并行映射（与 kernel 注释一致）：
    //     block = 256（scalar 版每线程一行）；
    //     grid = ceil(num_tokens / block)。
    //   调用要点：
    //     stream 从 ctx.stream 取；logits/tokens 的裸指针从各自
    //     Tensor::data() 解包并转成 kernel 参数类型；标量按签名顺序传
    //     （num_tokens, vocab_size）。
    const unsigned int block = 256;
    const unsigned int grid = (num_tokens - 1 + block) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    greedy_sample_kernel<<<grid, block, 0, s>>>(static_cast<const float*>(logits.data()),
                                                static_cast<int32_t*>(tokens->data()), num_tokens,
                                                vocab_size);
}

}  // namespace ccop
