#include "ccop/ops/prefill_attention.h"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ccop/cuda/error_cuda.h"

namespace ccop {
namespace {

template <typename T>
__device__ float cal_score(const T* q, const T* k, float scale, unsigned int head_dim) {
    float score = 0;
    for (unsigned int d = 0; d < head_dim; ++d) {
        score += static_cast<float>(q[d]) * static_cast<float>(k[d]);
    }
    return scale * score;
}

// PrefillAttention kernel 声明（接口已定，函数体待实现）。
// 模板参数：T（q/k_cache/v_cache/out 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版：每线程完整处理一个 (tq, qh) 输出行）。
//
// 数学形式（对每个请求 b、其每个 prompt token tq、每个 q 头 qh，GQA）：
//   kv_head = qh * num_kv_heads / num_q_heads（组内 q 头共享同一 kv 头）
//   prompt_len = query_start_loc[b+1] - query_start_loc[b]
//   prefix_len = context_lens[b] - prompt_len
//   p = tq - query_start_loc[b]（token 在本次 prompt 内的偏移）
//   causal：可见 KV token s ∈ [0, prefix_len + p + 1)（前缀 + 自己及之前）
//     逻辑块 i = s / block_size，块内偏移 r = s % block_size，
//     物理块 blk = block_table[b][i]
//     score[s] = scale * sum_d q[tq][qh][d] * k_cache[blk][r][kv_head][d]
//   m = max_s score[s]；l = sum_s exp(score[s] - m)
//   out[tq][qh][d] = sum_s (exp(score[s] - m) / l) * v_cache[blk][r][kv_head][d]
//
// 数据布局（全部行主序连续）：
//   q/out: [total_q_tokens, num_q_heads, head_dim]，
//       线性偏移 = (tq * num_q_heads + qh) * head_dim + d；
//   k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]，
//       线性偏移 = ((blk * block_size + r) * num_kv_heads + kv_head)
//                  * head_dim + d；
//   block_table: [batch_size, max_blocks_per_req]，线性偏移 = b * max_blocks + i；
//   query_start_loc: [batch_size + 1]（单调递增）；
//   context_lens: [batch_size]。
//
// 并行映射：
//   grid.x 与 block 内 threadIdx.x 一起平铺 total_q_tokens * num_q_heads
//   个输出行（block 向上取整，需越界守卫）；
//   blockIdx.x/threadIdx.x → 行索引 rid → tq = rid / num_q_heads、
//   qh = rid % num_q_heads；由 tq 在单调递增的 query_start_loc 中定位
//   所属请求 b（顺序扫或二分，任选）→ p = tq - query_start_loc[b]；
//   每线程对本请求循环 s ∈ [0, prefix_len + p + 1)（causal 上界含自身）。
//
// 数值注意：
//   - BF16 参与计算前提升到 float；点积与加权累加用 float 精度；
//   - softmax 数值稳定：exp 前必须减行内 max（score 可能为大正数）；
//     score 无静态存储可放（causal 长度运行时决定），各遍需要时按
//     (i, r) 定位物理块重新计算该元素（或 online-softmax，任选）；
//   - block_table 前 ceil(context_lens[b] / block_size) 个块的有效性由
//     调用方保证，kernel 不守卫；
//   - 维度与索引统一 unsigned int（uint32 上限约 42.9 亿，覆盖
//     32 头 × 100 万上下文 × head_dim 的目标场景）。
template <typename T>
__global__ void prefill_attention_kernel(const T* q, const T* k_cache, const T* v_cache,
                                         const int32_t* block_table, const int32_t* query_start_loc,
                                         const int32_t* context_lens, T* out, float scale,
                                         unsigned int batch_size, unsigned int total_q_tokens,
                                         unsigned int num_q_heads, unsigned int num_kv_heads,
                                         unsigned int head_dim, unsigned int block_size,
                                         unsigned int max_blocks_per_req) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_q_tokens * num_q_heads) {
        return;
    }
    unsigned int tq = tid / num_q_heads;
    unsigned int qh = tid % num_q_heads;
    unsigned int kvh = qh * num_kv_heads / num_q_heads;
    unsigned int b = 0;
    while (b + 1 < batch_size && query_start_loc[b + 1] <= tq) {
        ++b;
    }
    unsigned int prefix_len = context_lens[b] - (query_start_loc[b + 1] - query_start_loc[b]);
    unsigned int p = tq - query_start_loc[b];
    float m = -INFINITY;
    for (unsigned int s = 0; s < prefix_len + p + 1; ++s) {
        unsigned int i = s / block_size;
        unsigned int r = s % block_size;
        unsigned int blk = block_table[max_blocks_per_req * b + i];
        float score = cal_score<T>(
            q + (tq * num_q_heads + qh) * head_dim,
            k_cache + ((blk * block_size + r) * num_kv_heads + kvh) * head_dim, scale, head_dim);
        m = score > m ? score : m;
    }
    float l = 0;
    for (unsigned int s = 0; s < prefix_len + p + 1; ++s) {
        unsigned int i = s / block_size;
        unsigned int r = s % block_size;
        unsigned int blk = block_table[max_blocks_per_req * b + i];
        l += expf(cal_score<T>(q + (tq * num_q_heads + qh) * head_dim,
                               k_cache + ((blk * block_size + r) * num_kv_heads + kvh) * head_dim,
                               scale, head_dim) -
                  m);
    }
    for (unsigned int d = 0; d < head_dim; ++d) {
        float sum = 0;
        for (unsigned int s = 0; s < prefix_len + p + 1; ++s) {
            unsigned int i = s / block_size;
            unsigned int r = s % block_size;
            unsigned int blk = block_table[max_blocks_per_req * b + i];
            float p = expf(cal_score<T>(
                               q + (tq * num_q_heads + qh) * head_dim,
                               k_cache + ((blk * block_size + r) * num_kv_heads + kvh) * head_dim,
                               scale, head_dim) -
                           m) /
                      l;
            sum += p * static_cast<float>(
                           v_cache[((blk * block_size + r) * num_kv_heads + kvh) * head_dim + d]);
        }
        out[(tq * num_q_heads + qh) * head_dim + d] = static_cast<T>(sum);
    }
}

}  // namespace

Result<void> prefill_attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                               const Tensor& block_table, const Tensor& query_start_loc,
                               const Tensor& context_lens, Tensor* out, float scale,
                               const ExecutionContext& ctx) {
    if (!(q.valid() && k_cache.valid() && v_cache.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(q.rank() == 3 && q.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(k_cache.rank() == 4 && k_cache.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(v_cache.rank() == 4 && v_cache.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out != nullptr && out->valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->rank() == 3 && out->is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(q.dtype() == k_cache.dtype())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(q.dtype() == v_cache.dtype())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(q.dtype() == out->dtype())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(scale > 0.0f)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const unsigned int total_q_tokens = static_cast<unsigned int>(q.shape(0));
    const unsigned int num_q_heads = static_cast<unsigned int>(q.shape(1));
    const unsigned int num_kv_heads = static_cast<unsigned int>(k_cache.shape(2));
    const unsigned int head_dim = static_cast<unsigned int>(q.shape(2));
    const unsigned int block_size = static_cast<unsigned int>(k_cache.shape(1));
    if (!(total_q_tokens > 0 && num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(block_size > 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(num_q_heads % num_kv_heads == 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(k_cache.shape(0) > 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(k_cache.shape(3) == static_cast<std::int64_t>(head_dim))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(v_cache.shape(0) == k_cache.shape(0))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(v_cache.shape(1) == static_cast<std::int64_t>(block_size))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(v_cache.shape(2) == static_cast<std::int64_t>(num_kv_heads))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(v_cache.shape(3) == static_cast<std::int64_t>(head_dim))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->shape(0) == static_cast<std::int64_t>(total_q_tokens))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->shape(1) == static_cast<std::int64_t>(num_q_heads))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->shape(2) == static_cast<std::int64_t>(head_dim))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(block_table.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(block_table.rank() == 2 && block_table.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(block_table.dtype() == DType::kInt32)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const unsigned int batch_size = static_cast<unsigned int>(block_table.shape(0));
    if (!(batch_size > 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const unsigned int max_blocks_per_req = static_cast<unsigned int>(block_table.shape(1));
    if (!(max_blocks_per_req > 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(query_start_loc.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(query_start_loc.rank() == 1 && query_start_loc.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(query_start_loc.dtype() == DType::kInt32)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(query_start_loc.shape(0) == static_cast<std::int64_t>(batch_size + 1))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(context_lens.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(context_lens.rank() == 1 && context_lens.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(context_lens.dtype() == DType::kInt32)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(context_lens.shape(0) == static_cast<std::int64_t>(batch_size))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    //   并行映射（与 kernel 注释一致）：
    //     block = 256（scalar 版每线程一个输出行）；
    //     grid = ceil(total_q_tokens * num_q_heads / block)。
    //   调用要点：
    //     stream 从 ctx.stream 取；各数组裸指针从 Tensor::data() 解包并转成
    //     kernel 参数类型（block_table/query_start_loc/context_lens 为
    //     const int32_t*）；模板参数 T 按 dtype 取 __nv_bfloat16
    //     （kBFloat16）或 float（kFloat32）；标量按签名顺序传（scale,
    //     batch_size, total_q_tokens, num_q_heads, num_kv_heads, head_dim,
    //     block_size, max_blocks_per_req）。
    const unsigned int block = 256;
    const unsigned int grid = (total_q_tokens * num_q_heads - 1 + block) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    switch (q.dtype()) {
        case DType::kBFloat16:
            prefill_attention_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
                static_cast<const __nv_bfloat16*>(q.data()),
                static_cast<const __nv_bfloat16*>(k_cache.data()),
                static_cast<const __nv_bfloat16*>(v_cache.data()),
                static_cast<const int32_t*>(block_table.data()),
                static_cast<const int32_t*>(query_start_loc.data()),
                static_cast<const int32_t*>(context_lens.data()),
                static_cast<__nv_bfloat16*>(out->data()), scale, batch_size, total_q_tokens,
                num_q_heads, num_kv_heads, head_dim, block_size, max_blocks_per_req);
            return check_cuda_launch();
        case DType::kFloat32:
            prefill_attention_kernel<float><<<grid, block, 0, s>>>(
                static_cast<const float*>(q.data()), static_cast<const float*>(k_cache.data()),
                static_cast<const float*>(v_cache.data()),
                static_cast<const int32_t*>(block_table.data()),
                static_cast<const int32_t*>(query_start_loc.data()),
                static_cast<const int32_t*>(context_lens.data()), static_cast<float*>(out->data()),
                scale, batch_size, total_q_tokens, num_q_heads, num_kv_heads, head_dim, block_size,
                max_blocks_per_req);
            return check_cuda_launch();
        default:
            return std::unexpected(ErrorCode::kUnsupported);
    }
}

}  // namespace ccop
