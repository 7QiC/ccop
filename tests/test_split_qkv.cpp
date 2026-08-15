#include "ccop/ops/split_qkv.h"

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

// Host reference：按 Q/K/V 三段搬运（无算术，bit 精确）。
template <typename T>
void reference_split_qkv(const std::vector<T>& qkv, unsigned int num_tokens,
                         unsigned int num_q_heads, unsigned int num_kv_heads, unsigned int head_dim,
                         std::vector<T>& q, std::vector<T>& k, std::vector<T>& v) {
    const unsigned int q_dim = num_q_heads * head_dim;
    const unsigned int k_dim = num_kv_heads * head_dim;
    const unsigned int v_dim = num_kv_heads * head_dim;
    const unsigned int qkv_dim = q_dim + k_dim + v_dim;

    q.assign(static_cast<std::size_t>(num_tokens) * q_dim, T{});
    k.assign(static_cast<std::size_t>(num_tokens) * k_dim, T{});
    v.assign(static_cast<std::size_t>(num_tokens) * v_dim, T{});
    for (unsigned int t = 0; t < num_tokens; ++t) {
        const T* row = qkv.data() + static_cast<std::size_t>(t) * qkv_dim;
        for (unsigned int i = 0; i < q_dim; ++i) {
            q[static_cast<std::size_t>(t) * q_dim + i] = row[i];
        }
        for (unsigned int i = 0; i < k_dim; ++i) {
            k[static_cast<std::size_t>(t) * k_dim + i] = row[q_dim + i];
        }
        for (unsigned int i = 0; i < v_dim; ++i) {
            v[static_cast<std::size_t>(t) * v_dim + i] = row[q_dim + k_dim + i];
        }
    }
}

template <typename T>
void run_and_check(unsigned int num_tokens, unsigned int num_q_heads, unsigned int num_kv_heads,
                   unsigned int head_dim) {
    using Traits = FloatTraits<T>;
    const unsigned int q_dim = num_q_heads * head_dim;
    const unsigned int k_dim = num_kv_heads * head_dim;
    const unsigned int qkv_dim = q_dim + k_dim + k_dim;
    const std::size_t qkv_count = static_cast<std::size_t>(num_tokens) * qkv_dim;

    std::vector<T> qkv(qkv_count);
    for (std::size_t i = 0; i < qkv_count; ++i) {
        qkv[i] = Traits::from_float(-1.0f + 0.13f * static_cast<float>(i % 29));
    }

    std::vector<T> expected_q, expected_k, expected_v;
    reference_split_qkv(qkv, num_tokens, num_q_heads, num_kv_heads, head_dim, expected_q,
                        expected_k, expected_v);

    CudaMem qkv_mem(qkv_count * sizeof(T));
    CudaMem q_mem(expected_q.size() * sizeof(T));
    CudaMem k_mem(expected_k.size() * sizeof(T));
    CudaMem v_mem(expected_v.size() * sizeof(T));
    ASSERT_NE(qkv_mem.ptr_, nullptr);
    ASSERT_NE(q_mem.ptr_, nullptr);
    ASSERT_NE(k_mem.ptr_, nullptr);
    ASSERT_NE(v_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(qkv_mem.ptr_, qkv.data(), qkv_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    Tensor qkv_tensor(qkv_mem.ptr_, Traits::dtype, kCuda0,
                      {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(qkv_dim)});
    Tensor q_tensor(q_mem.ptr_, Traits::dtype, kCuda0,
                    {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_q_heads),
                     static_cast<std::int64_t>(head_dim)});
    Tensor k_tensor(k_mem.ptr_, Traits::dtype, kCuda0,
                    {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_kv_heads),
                     static_cast<std::int64_t>(head_dim)});
    Tensor v_tensor(v_mem.ptr_, Traits::dtype, kCuda0,
                    {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_kv_heads),
                     static_cast<std::int64_t>(head_dim)});

    split_qkv(qkv_tensor, &q_tensor, &k_tensor, &v_tensor, num_q_heads, num_kv_heads,
              ExecutionContext{});
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_q(expected_q.size());
    std::vector<T> host_k(expected_k.size());
    std::vector<T> host_v(expected_v.size());
    ASSERT_EQ(cudaMemcpy(host_q.data(), q_mem.ptr_, q_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_k.data(), k_mem.ptr_, k_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_v.data(), v_mem.ptr_, v_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected_q.size(); ++i) {
        EXPECT_EQ(Traits::to_float(host_q[i]), Traits::to_float(expected_q[i]));
    }
    for (std::size_t i = 0; i < expected_k.size(); ++i) {
        EXPECT_EQ(Traits::to_float(host_k[i]), Traits::to_float(expected_k[i]));
    }
    for (std::size_t i = 0; i < expected_v.size(); ++i) {
        EXPECT_EQ(Traits::to_float(host_v[i]), Traits::to_float(expected_v[i]));
    }
}

TEST(SplitQkvTest, Fp32Basic) { run_and_check<float>(2, 4, 2, 8); }

TEST(SplitQkvTest, Bf16Gqa) {
    // GQA 极端：nkv = 1。
    run_and_check<__nv_bfloat16>(2, 3, 1, 16);
}

TEST(SplitQkvTest, SingleToken) { run_and_check<float>(1, 2, 2, 4); }

TEST(SplitQkvTest, HeadDimOne) { run_and_check<float>(3, 2, 1, 1); }

TEST(SplitQkvTest, NonBlockMultiple) {
    // qkv 总元素数 7 * 64 = 448 非 block 整数倍：验证越界守卫。
    run_and_check<float>(7, 4, 2, 8);
}

}  // namespace
}  // namespace ccop
