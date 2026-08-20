#include "ccop/ops/rope.h"

#include <array>
#include <cmath>
#include <cstdint>
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

// Host reference cache，布局与 compute_rope_cache 相同：
//   cache[(pos * half_dim + pair) * 2 + 0] = cos，+1 = sin。
std::vector<float> reference_cache(int max_position, int rotary_dim, float rope_theta) {
    const int half_dim = rotary_dim / 2;
    std::vector<float> cache(static_cast<std::size_t>(max_position) * half_dim * 2);
    for (int pos = 0; pos < max_position; ++pos) {
        const float fpos = static_cast<float>(pos);
        for (int pair = 0; pair < half_dim; ++pair) {
            const float inv_freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * pair) /
                                                                   static_cast<float>(rotary_dim));
            const float angle = fpos * inv_freq;
            cache[static_cast<std::size_t>((pos * half_dim + pair) * 2) + 0] = std::cos(angle);
            cache[static_cast<std::size_t>((pos * half_dim + pair) * 2) + 1] = std::sin(angle);
        }
    }
    return cache;
}

// Host reference split-half RoPE，原地修改 q/k。
template <typename T>
void reference_rope(T* q, T* k, const int32_t* positions, const float* cache,
                    std::int64_t num_tokens, int num_q_heads, int num_kv_heads,
                    std::int64_t head_dim, int rotary_dim) {
    const int half_dim = rotary_dim / 2;
    const auto rotate = [&](T* x, int num_heads) {
        for (std::int64_t t = 0; t < num_tokens; ++t) {
            const int pos = positions[t];
            for (int h = 0; h < num_heads; ++h) {
                const std::int64_t base = (t * static_cast<std::int64_t>(num_heads) + h) * head_dim;
                for (int pair = 0; pair < half_dim; ++pair) {
                    const float cos_v =
                        cache[static_cast<std::size_t>((pos * half_dim + pair) * 2) + 0];
                    const float sin_v =
                        cache[static_cast<std::size_t>((pos * half_dim + pair) * 2) + 1];
                    const float x0 = FloatTraits<T>::to_float(x[base + pair]);
                    const float x1 = FloatTraits<T>::to_float(x[base + half_dim + pair]);
                    x[base + pair] = FloatTraits<T>::from_float(x0 * cos_v - x1 * sin_v);
                    x[base + half_dim + pair] = FloatTraits<T>::from_float(x1 * cos_v + x0 * sin_v);
                }
            }
        }
    };
    rotate(q, num_q_heads);
    rotate(k, num_kv_heads);
}

template <typename T>
void run_rope_and_check(const std::vector<float>& host_q, const std::vector<float>& host_k,
                        const std::vector<int32_t>& positions, const std::vector<float>& cache,
                        int num_q_heads, int num_kv_heads, std::int64_t head_dim, int rotary_dim,
                        float tolerance) {
    using Traits = FloatTraits<T>;
    const std::int64_t num_tokens = positions.size();
    const std::int64_t q_elems = static_cast<std::int64_t>(num_tokens) * num_q_heads * head_dim;
    const std::int64_t k_elems = static_cast<std::int64_t>(num_tokens) * num_kv_heads * head_dim;
    ASSERT_EQ(host_q.size(), static_cast<std::size_t>(q_elems));
    ASSERT_EQ(host_k.size(), static_cast<std::size_t>(k_elems));
    const std::int64_t half_dim = rotary_dim / 2;
    ASSERT_EQ(cache.size() % (static_cast<std::size_t>(half_dim) * 2), 0u);

    // Round inputs to the activation dtype, then compute the dtype-exact reference.
    std::vector<T> q_in(host_q.size());
    std::vector<T> k_in(host_k.size());
    for (std::size_t i = 0; i < host_q.size(); ++i) q_in[i] = Traits::from_float(host_q[i]);
    for (std::size_t i = 0; i < host_k.size(); ++i) k_in[i] = Traits::from_float(host_k[i]);
    std::vector<T> q_expected = q_in;
    std::vector<T> k_expected = k_in;
    reference_rope(q_expected.data(), k_expected.data(), positions.data(), cache.data(), num_tokens,
                   num_q_heads, num_kv_heads, head_dim, rotary_dim);

    CudaMem q_mem(q_in.size() * sizeof(T));
    CudaMem k_mem(k_in.size() * sizeof(T));
    CudaMem pos_mem(positions.size() * sizeof(int32_t));
    CudaMem cache_mem(cache.size() * sizeof(float));
    ASSERT_NE(q_mem.ptr_, nullptr);
    ASSERT_NE(k_mem.ptr_, nullptr);
    ASSERT_NE(pos_mem.ptr_, nullptr);
    ASSERT_NE(cache_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(q_mem.ptr_, q_in.data(), q_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(k_mem.ptr_, k_in.data(), k_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(pos_mem.ptr_, positions.data(), pos_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(cache_mem.ptr_, cache.data(), cache_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    const std::array<std::int64_t, kTensorMaxRank> q_shape{num_tokens, num_q_heads, head_dim};
    const std::array<std::int64_t, kTensorMaxRank> q_stride{
        static_cast<std::int64_t>(num_q_heads) * head_dim, head_dim, 1};
    const std::array<std::int64_t, kTensorMaxRank> k_shape{num_tokens, num_kv_heads, head_dim};
    const std::array<std::int64_t, kTensorMaxRank> k_stride{
        static_cast<std::int64_t>(num_kv_heads) * head_dim, head_dim, 1};
    const std::int64_t cache_max_position =
        static_cast<std::int64_t>(cache.size() / (static_cast<std::size_t>(half_dim) * 2));
    Tensor q_tensor(q_mem.ptr_, Traits::dtype, kCuda0, q_shape, q_stride, 3);
    Tensor k_tensor(k_mem.ptr_, Traits::dtype, kCuda0, k_shape, k_stride, 3);
    Tensor pos_tensor(pos_mem.ptr_, DType::kInt32, kCuda0, {num_tokens});
    Tensor cache_tensor(cache_mem.ptr_, DType::kFloat32, kCuda0, {cache_max_position, half_dim, 2});

    ASSERT_TRUE(
        rope(&q_tensor, &k_tensor, pos_tensor, cache_tensor, rotary_dim, ExecutionContext{}));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_q_out(q_in.size());
    std::vector<T> host_k_out(k_in.size());
    ASSERT_EQ(cudaMemcpy(host_q_out.data(), q_mem.ptr_, q_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_k_out.data(), k_mem.ptr_, k_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < q_expected.size(); ++i) {
        EXPECT_NEAR(Traits::to_float(host_q_out[i]), Traits::to_float(q_expected[i]), tolerance);
    }
    for (std::size_t i = 0; i < k_expected.size(); ++i) {
        EXPECT_NEAR(Traits::to_float(host_k_out[i]), Traits::to_float(k_expected[i]), tolerance);
    }
}

std::vector<float> make_input(std::int64_t n) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        v[static_cast<std::size_t>(i)] = 0.5f + 0.13f * static_cast<float>(i % 17);
    }
    return v;
}

TEST(RopeTest, Fp32Basic) {
    constexpr std::int64_t kNumTokens = 3;
    constexpr int kNumQHeads = 3;
    constexpr int kNumKvHeads = 3;
    constexpr std::int64_t kHeadDim = 8;
    constexpr int kRotaryDim = 8;
    constexpr float kTolerance = 1e-5f;
    const std::vector<int32_t> positions{0, 2, 5};
    const std::vector<float> cache = reference_cache(8, kRotaryDim, 10000.0f);

    const std::vector<float> q = make_input(kNumTokens * kNumQHeads * kHeadDim);
    const std::vector<float> k = make_input(kNumTokens * kNumKvHeads * kHeadDim);
    run_rope_and_check<float>(q, k, positions, cache, kNumQHeads, kNumKvHeads, kHeadDim, kRotaryDim,
                              kTolerance);
}

TEST(RopeTest, Fp32Gqa) {
    constexpr std::int64_t kNumTokens = 3;
    constexpr int kNumQHeads = 4;
    constexpr int kNumKvHeads = 2;
    constexpr std::int64_t kHeadDim = 8;
    constexpr int kRotaryDim = 8;
    constexpr float kTolerance = 1e-5f;
    const std::vector<int32_t> positions{0, 2, 5};
    const std::vector<float> cache = reference_cache(8, kRotaryDim, 10000.0f);

    const std::vector<float> q = make_input(kNumTokens * kNumQHeads * kHeadDim);
    const std::vector<float> k = make_input(kNumTokens * kNumKvHeads * kHeadDim);
    run_rope_and_check<float>(q, k, positions, cache, kNumQHeads, kNumKvHeads, kHeadDim, kRotaryDim,
                              kTolerance);
}

TEST(RopeTest, PartialRotary) {
    constexpr std::int64_t kNumTokens = 2;
    constexpr int kNumQHeads = 3;
    constexpr int kNumKvHeads = 1;
    constexpr std::int64_t kHeadDim = 8;
    constexpr int kRotaryDim = 4;
    constexpr float kTolerance = 1e-5f;
    const std::vector<int32_t> positions{1, 4};
    const std::vector<float> cache = reference_cache(6, kRotaryDim, 10000.0f);

    const std::vector<float> q = make_input(kNumTokens * kNumQHeads * kHeadDim);
    const std::vector<float> k = make_input(kNumTokens * kNumKvHeads * kHeadDim);
    run_rope_and_check<float>(q, k, positions, cache, kNumQHeads, kNumKvHeads, kHeadDim, kRotaryDim,
                              kTolerance);
}

TEST(RopeTest, NonContiguousPositions) {
    constexpr std::int64_t kNumTokens = 4;
    constexpr int kNumQHeads = 2;
    constexpr int kNumKvHeads = 2;
    constexpr std::int64_t kHeadDim = 16;
    constexpr int kRotaryDim = 16;
    constexpr float kTolerance = 1e-5f;
    const std::vector<int32_t> positions{7, 1, 15, 3};
    const std::vector<float> cache = reference_cache(16, kRotaryDim, 1000000.0f);

    const std::vector<float> q = make_input(kNumTokens * kNumQHeads * kHeadDim);
    const std::vector<float> k = make_input(kNumTokens * kNumKvHeads * kHeadDim);
    run_rope_and_check<float>(q, k, positions, cache, kNumQHeads, kNumKvHeads, kHeadDim, kRotaryDim,
                              kTolerance);
}

TEST(RopeTest, Bf16) {
    constexpr std::int64_t kNumTokens = 3;
    constexpr int kNumQHeads = 4;
    constexpr int kNumKvHeads = 2;
    constexpr std::int64_t kHeadDim = 8;
    constexpr int kRotaryDim = 8;
    constexpr float kTolerance = 1e-2f;
    const std::vector<int32_t> positions{0, 2, 5};
    const std::vector<float> cache = reference_cache(8, kRotaryDim, 10000.0f);

    const std::vector<float> q = make_input(kNumTokens * kNumQHeads * kHeadDim);
    const std::vector<float> k = make_input(kNumTokens * kNumKvHeads * kHeadDim);
    run_rope_and_check<__nv_bfloat16>(q, k, positions, cache, kNumQHeads, kNumKvHeads, kHeadDim,
                                      kRotaryDim, kTolerance);
}

TEST(RopeTest, IntegratedCache) {
    // 模拟框架流程：host 生成 cache → H2D → 构造 Tensor → rope。
    constexpr std::int64_t kNumTokens = 4;
    constexpr int kNumQHeads = 2;
    constexpr int kNumKvHeads = 1;
    constexpr std::int64_t kHeadDim = 16;
    constexpr int kRotaryDim = 16;
    constexpr int kMaxPosition = 16;
    constexpr float kTolerance = 1e-2f;
    const std::vector<int32_t> positions{3, 7, 1, 15};
    const std::vector<float> q = make_input(kNumTokens * kNumQHeads * kHeadDim);
    const std::vector<float> k = make_input(kNumTokens * kNumKvHeads * kHeadDim);

    const std::vector<float> gen_cache = reference_cache(kMaxPosition, kRotaryDim, 10000.0f);

    run_rope_and_check<float>(q, k, positions, gen_cache, kNumQHeads, kNumKvHeads, kHeadDim,
                              kRotaryDim, kTolerance);
}

TEST(RopeTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor q_tensor(ptr, DType::kFloat32, kCuda0, {2, 1, 4});
    Tensor k_tensor(ptr, DType::kFloat32, kCuda0, {1, 1, 4});
    Tensor pos_tensor(ptr, DType::kInt32, kCuda0, {2});
    Tensor cache_tensor(ptr, DType::kFloat32, kCuda0, {4, 2, 2});
    auto result = rope(&q_tensor, &k_tensor, pos_tensor, cache_tensor, 4, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(RopeTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor q_tensor(ptr, DType::kFloat16, kCuda0, {1, 1, 4});
    Tensor k_tensor(ptr, DType::kFloat16, kCuda0, {1, 1, 4});
    Tensor pos_tensor(ptr, DType::kInt32, kCuda0, {1});
    Tensor cache_tensor(ptr, DType::kFloat32, kCuda0, {4, 2, 2});
    auto result = rope(&q_tensor, &k_tensor, pos_tensor, cache_tensor, 4, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
