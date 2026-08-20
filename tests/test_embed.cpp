#include "ccop/ops/embed.h"

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

// Host reference：按 id 查表搬运（无算术，bit 精确），输入为 dtype 精确值。
template <typename T>
std::vector<T> reference_embed(const std::vector<T>& table,
                               const std::vector<std::int32_t>& token_ids, unsigned int d_model) {
    std::vector<T> out(token_ids.size() * d_model);
    for (std::size_t t = 0; t < token_ids.size(); ++t) {
        for (unsigned int d = 0; d < d_model; ++d) {
            out[t * d_model + d] = table[static_cast<std::size_t>(token_ids[t]) * d_model + d];
        }
    }
    return out;
}

template <typename T>
void run_and_check(const std::vector<float>& host_table, unsigned int vocab_size,
                   unsigned int d_model, const std::vector<std::int32_t>& token_ids) {
    using Traits = FloatTraits<T>;
    ASSERT_EQ(host_table.size(), static_cast<std::size_t>(vocab_size) * d_model);

    std::vector<T> table(host_table.size());
    for (std::size_t i = 0; i < host_table.size(); ++i) {
        table[i] = Traits::from_float(host_table[i]);
    }
    const std::vector<T> expected = reference_embed(table, token_ids, d_model);

    CudaMem table_mem(table.size() * sizeof(T));
    CudaMem ids_mem(token_ids.size() * sizeof(std::int32_t));
    CudaMem out_mem(expected.size() * sizeof(T));
    ASSERT_NE(table_mem.ptr_, nullptr);
    ASSERT_NE(ids_mem.ptr_, nullptr);
    ASSERT_NE(out_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(table_mem.ptr_, table.data(), table_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ids_mem.ptr_, token_ids.data(), ids_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    Tensor table_tensor(
        table_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(vocab_size), static_cast<std::int64_t>(d_model)});
    Tensor ids_tensor(ids_mem.ptr_, DType::kInt32, kCuda0,
                      {static_cast<std::int64_t>(token_ids.size())});
    Tensor out_tensor(
        out_mem.ptr_, Traits::dtype, kCuda0,
        {static_cast<std::int64_t>(token_ids.size()), static_cast<std::int64_t>(d_model)});

    ASSERT_TRUE(embed(table_tensor, ids_tensor, &out_tensor, ExecutionContext{}));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_out(expected.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), out_mem.ptr_, out_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(Traits::to_float(host_out[i]), Traits::to_float(expected[i]));
    }
}

std::vector<float> make_values(std::size_t n, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>(i % 11);
    }
    return v;
}

TEST(EmbedTest, Fp32Basic) {
    run_and_check<float>(make_values(8 * 6, -1.0f, 0.25f), 8, 6, {0, 4, 7});
}

TEST(EmbedTest, Bf16Basic) {
    run_and_check<__nv_bfloat16>(make_values(16 * 8, -0.5f, 0.125f), 16, 8, {15, 3});
}

TEST(EmbedTest, SingleToken) { run_and_check<float>(make_values(4 * 4, -1.0f, 0.5f), 4, 4, {2}); }

TEST(EmbedTest, DModelOne) {
    run_and_check<float>(make_values(5 * 1, -1.0f, 0.5f), 5, 1, {4, 0, 3});
}

TEST(EmbedTest, NonBlockMultiple) {
    // 总输出元素 9 * 32 = 288 非 block 整数倍：验证越界守卫。
    run_and_check<float>(make_values(32 * 32, -0.5f, 0.1f), 32, 32,
                         {0, 31, 5, 7, 12, 29, 3, 8, 17});
}

TEST(EmbedTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor table_tensor(ptr, DType::kFloat32, kCuda0, {4, 3});
    Tensor ids_tensor(ptr, DType::kInt32, kCuda0, {2});
    Tensor out_tensor(ptr, DType::kFloat32, kCuda0, {2, 4});
    auto result = embed(table_tensor, ids_tensor, &out_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(EmbedTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor table_tensor(ptr, DType::kFloat16, kCuda0, {4, 3});
    Tensor ids_tensor(ptr, DType::kInt32, kCuda0, {2});
    Tensor out_tensor(ptr, DType::kFloat16, kCuda0, {2, 3});
    auto result = embed(table_tensor, ids_tensor, &out_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
