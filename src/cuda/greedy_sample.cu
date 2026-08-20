#include "ccop/ops/greedy_sample.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "ccop/cuda/error_cuda.h"

namespace ccop {
namespace {

// GreedySample kernel：scalar 版，每线程一行。
// 无模板：logits 按采样上游约定固定 float32（不需要 dtype 分发）。
// 第一阶段（正确性为先，scalar 版：每线程一行）。
//
// 数学形式（按 logits_indices 逐行 argmax）：
//   row = logits_indices[b]
//   若 0 <= row < num_tokens：
//     tokens[b] = argmax_v logits[row][v]
//   否则跳过（padding/无效请求），tokens[b] 保持原值。
//   平局取最小 v（严格 > 才更新）。
//
// 数据布局（全部行主序连续）：
//   logits: [num_tokens, vocab_size]，线性偏移 = row * vocab_size + v；
//   logits_indices: [batch_size] int32，row = logits_indices[b]；
//   tokens: [batch_size] int32。
//
// 并行映射（batch 维度是输出维度）：
//   block = 256，grid.x = ceil(batch_size / block)；
//   b = blockIdx.x * blockDim.x + threadIdx.x。
//   第一个守卫：b >= batch_size 的尾块线程直接 return。
//   每个有效请求 b 读一次 row = logits_indices[b]，再遍历该 logits 行
//   vocab_size 个元素并维护 argmax。
//
// 越界语义（两个独立守卫，不要混用 batch 与 logits 行两个维度）：
//   - b 是 batch 维：只用于写 tokens[b] 和读 logits_indices[b]；
//   - row 是 logits 行维：0 <= row < num_tokens 才采样；
//     row < 0 或 row >= num_tokens 时直接 return，tokens[b] 保持原值。
//
// 数值注意：
//   - 纯比较无算术：平局语义用严格 > 更新，保证保留最小索引；
//   - logits 含 NaN 时行为未定义（调用方保证）。
__global__ void greedy_sample_kernel(const float* logits, const int32_t* logits_indices,
                                     int32_t* tokens, unsigned int batch_size,
                                     unsigned int num_tokens, unsigned int vocab_size) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= batch_size) {
        return;
    }
    int r = logits_indices[tid];
    if (r < 0 || r >= num_tokens) {
        return;
    }
    tokens[tid] = 0;
    float max = logits[r * vocab_size];
    for (unsigned int i = 1; i < vocab_size; ++i) {
        float tmp = logits[r * vocab_size + i];
        if (max < tmp) {
            max = tmp;
            tokens[tid] = i;
        }
    }
}

}  // namespace

Result<void> greedy_sample(const Tensor& logits, const Tensor& logits_indices, Tensor* tokens,
                           const ExecutionContext& ctx) {
    if (!(logits.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(logits.rank() == 2 && logits.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (logits.dtype() != DType::kFloat32) {
        return std::unexpected(ErrorCode::kUnsupported);
    }
    if (!(logits_indices.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(logits_indices.rank() == 1 && logits_indices.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(logits_indices.dtype() == DType::kInt32)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(tokens != nullptr && tokens->valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(tokens->rank() == 1 && tokens->is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(tokens->dtype() == DType::kInt32)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const unsigned int num_tokens = static_cast<unsigned int>(logits.shape(0));
    const unsigned int vocab_size = static_cast<unsigned int>(logits.shape(1));
    const unsigned int batch_size = static_cast<unsigned int>(logits_indices.shape(0));
    if (!(num_tokens > 0 && vocab_size > 0 && batch_size > 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(logits_indices.shape(0) == static_cast<std::int64_t>(batch_size))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(tokens->shape(0) == static_cast<std::int64_t>(batch_size))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }

    const unsigned int block = 256;
    const unsigned int grid = (batch_size - 1 + block) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    greedy_sample_kernel<<<grid, block, 0, s>>>(static_cast<const float*>(logits.data()),
                                                static_cast<const int32_t*>(logits_indices.data()),
                                                static_cast<int32_t*>(tokens->data()), batch_size,
                                                num_tokens, vocab_size);
    return check_cuda_launch();
}
}  // namespace ccop
