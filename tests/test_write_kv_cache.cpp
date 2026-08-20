#include "ccop/ops/write_kv_cache.h"

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

// Host reference：按 slot 散写（无算术，bit 精确）；越界 slot 的 token 跳过，
// cache 对应位置保持哨兵值。
template <typename T>
void reference_write_kv_cache(const std::vector<T>& k_new, const std::vector<T>& v_new,
                              const std::vector<std::int32_t>& slot_mapping, T k_sentinel,
                              T v_sentinel, unsigned int num_kv_heads, unsigned int head_dim,
                              unsigned int num_slots, std::vector<T>& k_cache,
                              std::vector<T>& v_cache) {
    const std::size_t cache_count = static_cast<std::size_t>(num_slots) * num_kv_heads * head_dim;
    k_cache.assign(cache_count, k_sentinel);
    v_cache.assign(cache_count, v_sentinel);

    for (std::size_t t = 0; t < slot_mapping.size(); ++t) {
        const std::int32_t slot = slot_mapping[t];
        if (slot < 0 || static_cast<unsigned int>(slot) >= num_slots) {
            continue;
        }
        for (unsigned int i = 0; i < num_kv_heads * head_dim; ++i) {
            k_cache[(static_cast<std::size_t>(slot) * num_kv_heads * head_dim) + i] =
                k_new[t * (num_kv_heads * head_dim) + i];
            v_cache[(static_cast<std::size_t>(slot) * num_kv_heads * head_dim) + i] =
                v_new[t * (num_kv_heads * head_dim) + i];
        }
    }
}

template <typename T>
void run_and_check(const std::vector<float>& host_k_new, const std::vector<float>& host_v_new,
                   const std::vector<std::int32_t>& slot_mapping, unsigned int num_kv_heads,
                   unsigned int head_dim, unsigned int num_slots) {
    using Traits = FloatTraits<T>;
    ASSERT_EQ(host_k_new.size(), host_v_new.size());
    const std::size_t num_tokens = slot_mapping.size();
    ASSERT_EQ(host_k_new.size(), num_tokens * static_cast<std::size_t>(num_kv_heads) * head_dim);

    std::vector<T> k_new(host_k_new.size());
    std::vector<T> v_new(host_v_new.size());
    for (std::size_t i = 0; i < host_k_new.size(); ++i) {
        k_new[i] = Traits::from_float(host_k_new[i]);
        v_new[i] = Traits::from_float(host_v_new[i]);
    }

    const T k_sentinel = Traits::from_float(-7.5f);
    const T v_sentinel = Traits::from_float(3.25f);
    std::vector<T> expected_k, expected_v;
    reference_write_kv_cache(k_new, v_new, slot_mapping, k_sentinel, v_sentinel, num_kv_heads,
                             head_dim, num_slots, expected_k, expected_v);

    CudaMem k_new_mem(k_new.size() * sizeof(T));
    CudaMem v_new_mem(v_new.size() * sizeof(T));
    CudaMem k_cache_mem(expected_k.size() * sizeof(T));
    CudaMem v_cache_mem(expected_v.size() * sizeof(T));
    CudaMem mapping_mem(slot_mapping.size() * sizeof(std::int32_t));
    ASSERT_NE(k_new_mem.ptr_, nullptr);
    ASSERT_NE(v_new_mem.ptr_, nullptr);
    ASSERT_NE(k_cache_mem.ptr_, nullptr);
    ASSERT_NE(v_cache_mem.ptr_, nullptr);
    ASSERT_NE(mapping_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(k_new_mem.ptr_, k_new.data(), k_new_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(v_new_mem.ptr_, v_new.data(), v_new_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    // cache 预填纯哨兵值：kernel 未写的 slot 必须保持哨兵（对拍能暴露漏写）。
    const std::vector<T> k_cache_init(expected_k.size(), k_sentinel);
    const std::vector<T> v_cache_init(expected_v.size(), v_sentinel);
    ASSERT_EQ(cudaMemcpy(k_cache_mem.ptr_, k_cache_init.data(), k_cache_mem.bytes_,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(v_cache_mem.ptr_, v_cache_init.data(), v_cache_mem.bytes_,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(mapping_mem.ptr_, slot_mapping.data(), mapping_mem.bytes_,
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    Tensor k_new_tensor(
        k_new_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_kv_heads),
         static_cast<std::int64_t>(head_dim)});
    Tensor v_new_tensor(
        v_new_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(num_kv_heads),
         static_cast<std::int64_t>(head_dim)});
    Tensor k_cache_tensor(
        k_cache_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(num_slots), static_cast<std::int64_t>(num_kv_heads),
         static_cast<std::int64_t>(head_dim)});
    Tensor v_cache_tensor(
        v_cache_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(num_slots), static_cast<std::int64_t>(num_kv_heads),
         static_cast<std::int64_t>(head_dim)});
    Tensor mapping_tensor(mapping_mem.ptr_, DType::kInt32, kCuda0,
                          {static_cast<std::int64_t>(num_tokens)});

    ASSERT_TRUE(write_kv_cache(k_new_tensor, v_new_tensor, &k_cache_tensor, &v_cache_tensor,
                               mapping_tensor, ExecutionContext{}));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_k_out(expected_k.size());
    std::vector<T> host_v_out(expected_v.size());
    ASSERT_EQ(
        cudaMemcpy(host_k_out.data(), k_cache_mem.ptr_, k_cache_mem.bytes_, cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(host_v_out.data(), v_cache_mem.ptr_, v_cache_mem.bytes_, cudaMemcpyDeviceToHost),
        cudaSuccess);

    for (std::size_t i = 0; i < expected_k.size(); ++i) {
        EXPECT_EQ(Traits::to_float(host_k_out[i]), Traits::to_float(expected_k[i]));
    }
    for (std::size_t i = 0; i < expected_v.size(); ++i) {
        EXPECT_EQ(Traits::to_float(host_v_out[i]), Traits::to_float(expected_v[i]));
    }
}

std::vector<float> make_values(std::size_t n, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>(i % 11);
    }
    return v;
}

TEST(WriteKvCacheTest, Fp32Basic) {
    const std::vector<float> k_new = make_values(2 * 2 * 8, -1.0f, 0.25f);
    const std::vector<float> v_new = make_values(2 * 2 * 8, 0.5f, -0.125f);
    run_and_check<float>(k_new, v_new, {0, 3}, 2, 8, 4);
}

TEST(WriteKvCacheTest, Bf16WithPadding) {
    // slot_mapping 含 -1（padding）与 >= num_slots 的越界值：对应 token 不写入。
    const std::vector<float> k_new = make_values(4 * 2 * 8, -1.0f, 0.25f);
    const std::vector<float> v_new = make_values(4 * 2 * 8, 0.5f, -0.125f);
    run_and_check<__nv_bfloat16>(k_new, v_new, {1, -1, 4, 0}, 2, 8, 4);
}

TEST(WriteKvCacheTest, SingleToken) {
    const std::vector<float> k_new = make_values(1 * 2 * 4, 0.0f, 0.5f);
    const std::vector<float> v_new = make_values(1 * 2 * 4, 1.0f, 0.25f);
    run_and_check<float>(k_new, v_new, {0}, 2, 4, 2);
}

TEST(WriteKvCacheTest, ScatterOrder) {
    // 乱序全覆盖：slot 顺序与 token 顺序不同，所有 cache 位置都被写入。
    const std::vector<float> k_new = make_values(4 * 2 * 8, -1.0f, 0.25f);
    const std::vector<float> v_new = make_values(4 * 2 * 8, 0.5f, -0.125f);
    run_and_check<float>(k_new, v_new, {2, 0, 3, 1}, 2, 8, 4);
}

TEST(WriteKvCacheTest, HeadDimOne) {
    const std::vector<float> k_new = make_values(3 * 2 * 1, -2.0f, 0.5f);
    const std::vector<float> v_new = make_values(3 * 2 * 1, 2.0f, -0.5f);
    run_and_check<float>(k_new, v_new, {0, 2, 1}, 2, 1, 3);
}

TEST(WriteKvCacheTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor k_new_tensor(ptr, DType::kFloat32, kCuda0, {2, 1, 2});
    Tensor v_new_tensor(ptr, DType::kFloat32, kCuda0, {2, 1, 2});
    Tensor k_cache_tensor(ptr, DType::kFloat32, kCuda0, {4, 1, 2});
    Tensor v_cache_tensor(ptr, DType::kFloat32, kCuda0, {4, 1, 2});
    Tensor mapping_tensor(ptr, DType::kInt32, kCuda0, {3});
    auto result = write_kv_cache(k_new_tensor, v_new_tensor, &k_cache_tensor, &v_cache_tensor,
                                 mapping_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(WriteKvCacheTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor k_new_tensor(ptr, DType::kFloat16, kCuda0, {2, 1, 2});
    Tensor v_new_tensor(ptr, DType::kFloat16, kCuda0, {2, 1, 2});
    Tensor k_cache_tensor(ptr, DType::kFloat16, kCuda0, {4, 1, 2});
    Tensor v_cache_tensor(ptr, DType::kFloat16, kCuda0, {4, 1, 2});
    Tensor mapping_tensor(ptr, DType::kInt32, kCuda0, {2});
    auto result = write_kv_cache(k_new_tensor, v_new_tensor, &k_cache_tensor, &v_cache_tensor,
                                 mapping_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
