#include "ccop/ops/naive_attention.h"

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

// NaiveAttention kernel：causal 语义，scalar 版每线程完整处理一个 (t, qh) 输出行。
// 模板参数：T（q/k/v/out 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版：每线程完整处理一个 (t, qh) 输出行）。
//
// 数学形式（对每个 token t、每个 q 头 qh，GQA）：
//   kv_head = qh * num_kv_heads / num_q_heads（组内 q 头共享同一 kv 头）
//   score[s] = scale * sum_d q[t][qh][d] * k[s][kv_head][d]   （s = 0..t）
//   m = max_s score[s]；l = sum_s exp(score[s] - m)
//   p[s] = exp(score[s] - m) / l
//   out[t][qh][d] = sum_s p[s] * v[s][kv_head][d]             （d = 0..hd-1）
//
// 数据布局（全部行主序连续）：
//   q/out: [num_tokens, num_q_heads, head_dim]，
//       线性偏移 = (t * num_q_heads + qh) * head_dim + d；
//   k/v: [num_tokens, num_kv_heads, head_dim]，
//       线性偏移 = (s * num_kv_heads + kv_head) * head_dim + d。
//
// 并行映射：
//   grid.x 与 block 内 threadIdx.x 一起平铺 num_tokens * num_q_heads 个
//   输出行（block 向上取整，需越界守卫）；
//   blockIdx.x/threadIdx.x → 行索引 r → t = r / num_q_heads、
//   qh = r % num_q_heads；每线程对该行循环 s ∈ [0, t]（causal 上界含自身）。
//
// 数值注意：
//   - BF16 参与计算前提升到 float；点积与加权累加用 float 精度；
//   - softmax 数值稳定：exp 前必须减行内 max（score 可能为大正数），
//     即先求 m 再算 exp(score - m)；score 无静态存储可放（seq_len 运行时
//     决定），各遍需要时重新计算该元素，或按 online-softmax 增量维护
//     m/l（两种途径任选，注意 l 的增量更新形式要保持数值稳定）；
//   - 维度与索引统一 unsigned int（uint32 上限约 42.9 亿，覆盖
//     32 头 × 100 万上下文 × head_dim 的目标场景）。
template <typename T>
__global__ void naive_attention_kernel(const T* q, const T* k, const T* v, T* out, float scale,
                                       unsigned int num_tokens, unsigned int num_q_heads,
                                       unsigned int num_kv_heads, unsigned int head_dim) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_tokens * num_q_heads) {
        return;
    }
    unsigned int t = tid / num_q_heads;
    unsigned int qh = tid % num_q_heads;
    unsigned int kvh = qh * num_kv_heads / num_q_heads;
    float max_s = -INFINITY;
    for (unsigned int s = 0; s <= t; ++s) {
        float score = cal_score<T>(q + (t * num_q_heads + qh) * head_dim,
                                   k + (s * num_kv_heads + kvh) * head_dim, scale, head_dim);
        max_s = max_s < score ? score : max_s;
    }
    float sum_es = 0;
    for (unsigned int s = 0; s <= t; ++s) {
        sum_es += expf(cal_score<T>(q + (t * num_q_heads + qh) * head_dim,
                                    k + (s * num_kv_heads + kvh) * head_dim, scale, head_dim) -
                       max_s);
    }
    for (unsigned int d = 0; d < head_dim; ++d) {
        float sum = 0;
        for (unsigned int s = 0; s <= t; ++s) {
            float p = expf(cal_score<T>(q + (t * num_q_heads + qh) * head_dim,
                                        k + (s * num_kv_heads + kvh) * head_dim, scale, head_dim) -
                           max_s) /
                      sum_es;
            sum += p * static_cast<float>(v[(s * num_kv_heads + kvh) * head_dim + d]);
        }
        out[(t * num_q_heads + qh) * head_dim + d] = static_cast<T>(sum);
    }
}

}  // namespace

Result<void> naive_attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor* out,
                             float scale, const ExecutionContext& ctx) {
    if (!(q.valid() && k.valid() && v.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(q.rank() == 3 && q.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(k.rank() == 3 && k.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(v.rank() == 3 && v.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out != nullptr && out->valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->rank() == 3 && out->is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(q.dtype() == k.dtype())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(q.dtype() == v.dtype())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(q.dtype() == out->dtype())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(scale > 0.0f)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const unsigned int num_tokens = static_cast<unsigned int>(q.shape(0));
    const unsigned int num_q_heads = static_cast<unsigned int>(q.shape(1));
    const unsigned int num_kv_heads = static_cast<unsigned int>(k.shape(1));
    const unsigned int head_dim = static_cast<unsigned int>(q.shape(2));
    if (!(num_tokens > 0 && num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(num_q_heads % num_kv_heads == 0)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(k.shape(0) == static_cast<std::int64_t>(num_tokens))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(v.shape(0) == static_cast<std::int64_t>(num_tokens))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(k.shape(2) == static_cast<std::int64_t>(head_dim))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(v.shape(2) == static_cast<std::int64_t>(head_dim))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->shape(0) == static_cast<std::int64_t>(num_tokens))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->shape(1) == static_cast<std::int64_t>(num_q_heads))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->shape(2) == static_cast<std::int64_t>(head_dim))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    //   并行映射（与 kernel 注释一致）：
    //     block = 256（scalar 版每线程一个输出行）；
    //     grid.x = ceil(num_tokens * num_q_heads / block)。
    //   调用要点：
    //     stream 从 ctx.stream 取；各数组裸指针从 Tensor::data() 解包并转成
    //     kernel 参数类型；模板参数 T 按 dtype 取 __nv_bfloat16（kBFloat16）
    //     或 float（kFloat32）；标量按签名顺序传（scale, num_tokens,
    //     num_q_heads, num_kv_heads, head_dim）。
    const unsigned int block = 256;
    const unsigned int grid = (num_tokens * num_q_heads - 1 + block) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    switch (q.dtype()) {
        case DType::kBFloat16:
            naive_attention_kernel<__nv_bfloat16>
                <<<grid, block, 0, s>>>(static_cast<const __nv_bfloat16*>(q.data()),
                                        static_cast<const __nv_bfloat16*>(k.data()),
                                        static_cast<const __nv_bfloat16*>(v.data()),
                                        static_cast<__nv_bfloat16*>(out->data()), scale, num_tokens,
                                        num_q_heads, num_kv_heads, head_dim);
            return check_cuda_launch();
        case DType::kFloat32:
            naive_attention_kernel<float><<<grid, block, 0, s>>>(
                static_cast<const float*>(q.data()), static_cast<const float*>(k.data()),
                static_cast<const float*>(v.data()), static_cast<float*>(out->data()), scale,
                num_tokens, num_q_heads, num_kv_heads, head_dim);
            return check_cuda_launch();
        default:
            return std::unexpected(ErrorCode::kUnsupported);
    }
}

}  // namespace ccop
