#include "ccop/ops/decode_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "ccop/tensor.h"

namespace ccop {
namespace {

constexpr Device kCuda0{DeviceType::kCUDA, 0};

struct CudaMem {
    explicit CudaMem(std::size_t bytes) : bytes_(bytes) { cudaMalloc(&ptr_, bytes); }
    ~CudaMem() {
        if (ptr_) cudaFree(ptr_);
    }
    CudaMem(const CudaMem&) = delete;
    CudaMem& operator=(const CudaMem&) = delete;

    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

template <typename T>
struct FloatTraits;

template <>
struct FloatTraits<float> {
    static constexpr DType dtype = DType::kFloat32;
    static float from_float(float v) { return v; }
    static float to_float(float v) { return v; }
};

template <>
struct FloatTraits<__nv_bfloat16> {
    static constexpr DType dtype = DType::kBFloat16;
    static __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
    static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
};

// Host reference：paged 读取 + 数值稳定 softmax + GQA，输入为 dtype 精确值。
// blk < 0 的 padding 块不参与 score/求和。
template <typename T>
std::vector<T> reference_decode_attention(
    const std::vector<T>& q, const std::vector<T>& k_cache, const std::vector<T>& v_cache,
    const std::vector<std::int32_t>& block_table, const std::vector<std::int32_t>& context_lens,
    float scale, unsigned int batch_size, unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int head_dim, unsigned int block_size, unsigned int max_blocks_per_req) {
    using Traits = FloatTraits<T>;
    std::vector<T> out(static_cast<std::size_t>(batch_size) * num_q_heads * head_dim);
    const unsigned int group = num_q_heads / num_kv_heads;
    for (unsigned int b = 0; b < batch_size; ++b) {
        const unsigned int kv_len = static_cast<unsigned int>(context_lens[b]);
        for (unsigned int qh = 0; qh < num_q_heads; ++qh) {
            const unsigned int kv = qh / group;
            float m = -std::numeric_limits<float>::infinity();
            std::vector<float> score(kv_len, 0.0f);
            for (unsigned int s = 0; s < kv_len; ++s) {
                const unsigned int i = s / block_size;
                const unsigned int r = s % block_size;
                const std::int32_t blk = block_table[b * max_blocks_per_req + i];
                if (blk < 0) {
                    score[s] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float acc = 0.0f;
                for (unsigned int d = 0; d < head_dim; ++d) {
                    acc +=
                        Traits::to_float(
                            q[(static_cast<std::size_t>(b) * num_q_heads + qh) * head_dim + d]) *
                        Traits::to_float(k_cache[((static_cast<std::size_t>(blk) * block_size + r) *
                                                      num_kv_heads +
                                                  kv) *
                                                     head_dim +
                                                 d]);
                }
                score[s] = acc * scale;
                m = std::max(m, score[s]);
            }
            float l = 0.0f;
            for (unsigned int s = 0; s < kv_len; ++s) {
                if (block_table[b * max_blocks_per_req + s / block_size] < 0) {
                    continue;
                }
                l += std::exp(score[s] - m);
            }
            for (unsigned int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (unsigned int s = 0; s < kv_len; ++s) {
                    const unsigned int i = s / block_size;
                    const unsigned int r = s % block_size;
                    const std::int32_t blk = block_table[b * max_blocks_per_req + i];
                    if (blk < 0) {
                        continue;
                    }
                    acc +=
                        std::exp(score[s] - m) / l *
                        Traits::to_float(v_cache[((static_cast<std::size_t>(blk) * block_size + r) *
                                                      num_kv_heads +
                                                  kv) *
                                                     head_dim +
                                                 d]);
                }
                out[(static_cast<std::size_t>(b) * num_q_heads + qh) * head_dim + d] =
                    Traits::from_float(acc);
            }
        }
    }
    return out;
}

template <typename T>
void run_and_check(const std::vector<float>& host_q, const std::vector<float>& host_k_cache,
                   const std::vector<float>& host_v_cache,
                   const std::vector<std::int32_t>& block_table,
                   const std::vector<std::int32_t>& context_lens, unsigned int batch_size,
                   unsigned int num_q_heads, unsigned int num_kv_heads, unsigned int head_dim,
                   unsigned int num_blocks, unsigned int block_size, float tolerance) {
    using Traits = FloatTraits<T>;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const unsigned int max_blocks_per_req =
        static_cast<unsigned int>(block_table.size()) / batch_size;
    ASSERT_EQ(host_q.size(), static_cast<std::size_t>(batch_size) * num_q_heads * head_dim);
    ASSERT_EQ(host_k_cache.size(),
              static_cast<std::size_t>(num_blocks) * block_size * num_kv_heads * head_dim);
    ASSERT_EQ(host_v_cache.size(), host_k_cache.size());
    ASSERT_EQ(context_lens.size(), batch_size);

    std::vector<T> q(host_q.size());
    std::vector<T> k_cache(host_k_cache.size());
    std::vector<T> v_cache(host_v_cache.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = Traits::from_float(host_q[i]);
    }
    for (std::size_t i = 0; i < k_cache.size(); ++i) {
        k_cache[i] = Traits::from_float(host_k_cache[i]);
        v_cache[i] = Traits::from_float(host_v_cache[i]);
    }
    const std::vector<T> expected = reference_decode_attention(
        q, k_cache, v_cache, block_table, context_lens, scale, batch_size, num_q_heads,
        num_kv_heads, head_dim, block_size, max_blocks_per_req);

    CudaMem q_mem(q.size() * sizeof(T));
    CudaMem k_cache_mem(k_cache.size() * sizeof(T));
    CudaMem v_cache_mem(v_cache.size() * sizeof(T));
    CudaMem table_mem(block_table.size() * sizeof(std::int32_t));
    CudaMem lens_mem(context_lens.size() * sizeof(std::int32_t));
    CudaMem out_mem(expected.size() * sizeof(T));
    ASSERT_NE(q_mem.ptr_, nullptr);
    ASSERT_NE(k_cache_mem.ptr_, nullptr);
    ASSERT_NE(v_cache_mem.ptr_, nullptr);
    ASSERT_NE(table_mem.ptr_, nullptr);
    ASSERT_NE(lens_mem.ptr_, nullptr);
    ASSERT_NE(out_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(q_mem.ptr_, q.data(), q_mem.bytes_, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(k_cache_mem.ptr_, k_cache.data(), k_cache_mem.bytes_, cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(v_cache_mem.ptr_, v_cache.data(), v_cache_mem.bytes_, cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(table_mem.ptr_, block_table.data(), table_mem.bytes_, cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(lens_mem.ptr_, context_lens.data(), lens_mem.bytes_, cudaMemcpyHostToDevice),
        cudaSuccess);

    Tensor q_tensor(q_mem.ptr_, Traits::dtype, kCuda0,
                    {static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(num_q_heads),
                     static_cast<std::int64_t>(head_dim)});
    Tensor k_cache_tensor(
        k_cache_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(num_blocks), static_cast<std::int64_t>(block_size),
         static_cast<std::int64_t>(num_kv_heads), static_cast<std::int64_t>(head_dim)});
    Tensor v_cache_tensor(
        v_cache_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(num_blocks), static_cast<std::int64_t>(block_size),
         static_cast<std::int64_t>(num_kv_heads), static_cast<std::int64_t>(head_dim)});
    Tensor table_tensor(
        table_mem.ptr_, DType::kInt32, kCuda0,
        {static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(max_blocks_per_req)});
    Tensor lens_tensor(lens_mem.ptr_, DType::kInt32, kCuda0,
                       {static_cast<std::int64_t>(batch_size)});
    Tensor out_tensor(
        out_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(num_q_heads),
         static_cast<std::int64_t>(head_dim)});

    decode_attention(q_tensor, k_cache_tensor, v_cache_tensor, table_tensor, lens_tensor,
                     &out_tensor, scale, ExecutionContext{});
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_out(expected.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), out_mem.ptr_, out_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(Traits::to_float(host_out[i]), Traits::to_float(expected[i]), tolerance);
    }
}

std::vector<float> make_values(std::size_t n, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>(i % 13);
    }
    return v;
}

TEST(DecodeAttentionTest, Fp32Basic) {
    // batch=2：req0 ctx_len=5（块 [0,2]），req1 ctx_len=3（块 [1,-1] padding）。
    run_and_check<float>(
        make_values(2 * 2 * 8, -0.5f, 0.2f), make_values(4 * 4 * 2 * 8, -0.3f, 0.15f),
        make_values(4 * 4 * 2 * 8, -0.7f, 0.25f), {0, 2, 1, -1}, {5, 3}, 2, 2, 2, 8, 4, 4, 1e-4f);
}

TEST(DecodeAttentionTest, Bf16Gqa) {
    // GQA：4 q 头共享 2 kv 头；req0 ctx_len=5 占 3 块，req1 ctx_len=2 占 1 块。
    run_and_check<__nv_bfloat16>(make_values(2 * 4 * 16, -0.5f, 0.2f),
                                 make_values(3 * 2 * 2 * 16, -0.3f, 0.15f),
                                 make_values(3 * 2 * 2 * 16, -0.7f, 0.25f), {0, 1, 2, 1, -1, -1},
                                 {5, 2}, 2, 4, 2, 16, 3, 2, 1e-2f);
}

TEST(DecodeAttentionTest, SingleRequest) {
    // 单请求、单 KV token：softmax 恒为 1，out = v 直接搬运。
    run_and_check<float>(
        make_values(1 * 2 * 4, -0.5f, 0.2f), make_values(2 * 4 * 2 * 4, -0.3f, 0.15f),
        make_values(2 * 4 * 2 * 4, -0.7f, 0.25f), {0, -1}, {1}, 1, 2, 2, 4, 2, 4, 1e-4f);
}

TEST(DecodeAttentionTest, LargeScores) {
    // 大 score 验证数值稳定：q/k 全 10、hd=8 → score≈800，须 max 减法。
    const std::vector<float> q(2 * 2 * 8, 10.0f);
    const std::vector<float> k_cache(4 * 4 * 2 * 8, 10.0f);
    const std::vector<float> v_cache(4 * 4 * 2 * 8, 1.0f);
    run_and_check<float>(q, k_cache, v_cache, {0, 2, 1, -1}, {5, 3}, 2, 2, 2, 8, 4, 4, 1e-4f);
}

}  // namespace
}  // namespace ccop
