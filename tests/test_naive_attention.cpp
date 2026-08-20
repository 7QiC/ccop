#include "ccop/ops/naive_attention.h"

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

// Host reference：causal naive attention（float 域、数值稳定 softmax），输入为 dtype 精确值。
template <typename T>
std::vector<T> reference_naive_attention(const std::vector<T>& q, const std::vector<T>& k,
                                         const std::vector<T>& v, float scale,
                                         unsigned int num_tokens, unsigned int num_q_heads,
                                         unsigned int num_kv_heads, unsigned int head_dim) {
    using Traits = FloatTraits<T>;
    std::vector<T> out(q.size());
    const unsigned int group = num_q_heads / num_kv_heads;
    for (unsigned int t = 0; t < num_tokens; ++t) {
        for (unsigned int qh = 0; qh < num_q_heads; ++qh) {
            const unsigned int kv = qh / group;
            std::vector<float> score(num_tokens);
            float m = -std::numeric_limits<float>::infinity();
            for (unsigned int s = 0; s <= t; ++s) {
                float acc = 0.0f;
                for (unsigned int d = 0; d < head_dim; ++d) {
                    acc += Traits::to_float(
                               q[(static_cast<std::size_t>(t) * num_q_heads + qh) * head_dim + d]) *
                           Traits::to_float(
                               k[(static_cast<std::size_t>(s) * num_kv_heads + kv) * head_dim + d]);
                }
                score[s] = acc * scale;
                m = std::max(m, score[s]);
            }
            float l = 0.0f;
            for (unsigned int s = 0; s <= t; ++s) {
                l += std::exp(score[s] - m);
            }
            for (unsigned int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (unsigned int s = 0; s <= t; ++s) {
                    acc += std::exp(score[s] - m) / l *
                           Traits::to_float(
                               v[(static_cast<std::size_t>(s) * num_kv_heads + kv) * head_dim + d]);
                }
                out[(static_cast<std::size_t>(t) * num_q_heads + qh) * head_dim + d] =
                    Traits::from_float(acc);
            }
        }
    }
    return out;
}

template <typename T>
void run_and_check(const std::vector<float>& host_q, const std::vector<float>& host_k,
                   const std::vector<float>& host_v, unsigned int num_tokens,
                   unsigned int num_q_heads, unsigned int num_kv_heads, unsigned int head_dim,
                   float tolerance) {
    using Traits = FloatTraits<T>;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ASSERT_EQ(host_q.size(), static_cast<std::size_t>(num_tokens) * num_q_heads * head_dim);
    ASSERT_EQ(host_k.size(), static_cast<std::size_t>(num_tokens) * num_kv_heads * head_dim);
    ASSERT_EQ(host_v.size(), host_k.size());

    std::vector<T> q(host_q.size());
    std::vector<T> k(host_k.size());
    std::vector<T> v(host_v.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = Traits::from_float(host_q[i]);
    }
    for (std::size_t i = 0; i < k.size(); ++i) {
        k[i] = Traits::from_float(host_k[i]);
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = Traits::from_float(host_v[i]);
    }
    const std::vector<T> expected =
        reference_naive_attention(q, k, v, scale, num_tokens, num_q_heads, num_kv_heads, head_dim);

    CudaMem q_mem(q.size() * sizeof(T));
    CudaMem k_mem(k.size() * sizeof(T));
    CudaMem v_mem(v.size() * sizeof(T));
    CudaMem out_mem(expected.size() * sizeof(T));
    ASSERT_NE(q_mem.ptr_, nullptr);
    ASSERT_NE(k_mem.ptr_, nullptr);
    ASSERT_NE(v_mem.ptr_, nullptr);
    ASSERT_NE(out_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(q_mem.ptr_, q.data(), q_mem.bytes_, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(k_mem.ptr_, k.data(), k_mem.bytes_, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(v_mem.ptr_, v.data(), v_mem.bytes_, cudaMemcpyHostToDevice), cudaSuccess);

    Tensor q_tensor(q_mem.ptr_, Traits::dtype, kCuda0,
                    {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_q_heads),
                     static_cast<std::int64_t>(head_dim)});
    Tensor k_tensor(k_mem.ptr_, Traits::dtype, kCuda0,
                    {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_kv_heads),
                     static_cast<std::int64_t>(head_dim)});
    Tensor v_tensor(v_mem.ptr_, Traits::dtype, kCuda0,
                    {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_kv_heads),
                     static_cast<std::int64_t>(head_dim)});
    Tensor out_tensor(
        out_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_q_heads),
         static_cast<std::int64_t>(head_dim)});

    ASSERT_TRUE(
        naive_attention(q_tensor, k_tensor, v_tensor, &out_tensor, scale, ExecutionContext{}));
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

TEST(NaiveAttentionTest, Fp32Basic) {
    run_and_check<float>(make_values(4 * 2 * 8, -0.5f, 0.2f), make_values(4 * 2 * 8, -0.3f, 0.15f),
                         make_values(4 * 2 * 8, -0.7f, 0.25f), 4, 2, 2, 8, 1e-4f);
}

TEST(NaiveAttentionTest, Fp32Gqa) {
    // GQA：4 个 q 头共享 2 个 kv 头（组大小 2）。
    run_and_check<float>(make_values(3 * 4 * 8, -0.5f, 0.2f), make_values(3 * 2 * 8, -0.3f, 0.15f),
                         make_values(3 * 2 * 8, -0.7f, 0.25f), 3, 4, 2, 8, 1e-4f);
}

TEST(NaiveAttentionTest, Bf16Basic) {
    run_and_check<__nv_bfloat16>(make_values(3 * 2 * 16, -0.5f, 0.2f),
                                 make_values(3 * 2 * 16, -0.3f, 0.15f),
                                 make_values(3 * 2 * 16, -0.7f, 0.25f), 3, 2, 2, 16, 1e-2f);
}

TEST(NaiveAttentionTest, SingleToken) {
    run_and_check<float>(make_values(1 * 2 * 4, -0.5f, 0.2f), make_values(1 * 1 * 4, -0.3f, 0.15f),
                         make_values(1 * 1 * 4, -0.7f, 0.25f), 1, 2, 1, 4, 1e-4f);
}

TEST(NaiveAttentionTest, LargeScores) {
    // 大 score 验证数值稳定：q/k 元素均 ~10、hd=8 → score ~800，
    // 不做 max 减法的实现会 exp 溢出为 inf → 0/0 NaN。
    const std::vector<float> q(2 * 2 * 8, 10.0f);
    const std::vector<float> k(2 * 2 * 8, 10.0f);
    const std::vector<float> v(2 * 2 * 8, 1.0f);
    run_and_check<float>(q, k, v, 2, 2, 2, 8, 1e-4f);
}

TEST(NaiveAttentionTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor q_tensor(ptr, DType::kFloat32, kCuda0, {1, 3, 2});
    Tensor k_tensor(ptr, DType::kFloat32, kCuda0, {1, 2, 2});
    Tensor v_tensor(ptr, DType::kFloat32, kCuda0, {1, 2, 2});
    Tensor out_tensor(ptr, DType::kFloat32, kCuda0, {1, 3, 2});
    auto result =
        naive_attention(q_tensor, k_tensor, v_tensor, &out_tensor, 0.5f, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(NaiveAttentionTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor q_tensor(ptr, DType::kFloat16, kCuda0, {1, 2, 2});
    Tensor k_tensor(ptr, DType::kFloat16, kCuda0, {1, 1, 2});
    Tensor v_tensor(ptr, DType::kFloat16, kCuda0, {1, 1, 2});
    Tensor out_tensor(ptr, DType::kFloat16, kCuda0, {1, 2, 2});
    auto result =
        naive_attention(q_tensor, k_tensor, v_tensor, &out_tensor, 0.5f, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
