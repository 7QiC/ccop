#include "ccop/ops/element_add.h"

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

// Host reference: out[i] = dst0[i] + src[i]，输入为 dtype 精确值，float 域计算。
template <typename T>
std::vector<T> reference_element_add(const std::vector<T>& dst0, const std::vector<T>& src) {
    using Traits = FloatTraits<T>;
    std::vector<T> out(dst0.size());
    for (std::size_t i = 0; i < dst0.size(); ++i) {
        out[i] = Traits::from_float(Traits::to_float(dst0[i]) + Traits::to_float(src[i]));
    }
    return out;
}

template <typename T>
void run_and_check(const std::vector<float>& host_dst0, const std::vector<float>& host_src,
                   float tolerance) {
    using Traits = FloatTraits<T>;
    ASSERT_EQ(host_dst0.size(), host_src.size());

    std::vector<T> dst0(host_dst0.size());
    std::vector<T> src(host_src.size());
    for (std::size_t i = 0; i < host_dst0.size(); ++i) {
        dst0[i] = Traits::from_float(host_dst0[i]);
        src[i] = Traits::from_float(host_src[i]);
    }
    const std::vector<T> expected = reference_element_add(dst0, src);

    CudaMem dst_mem(dst0.size() * sizeof(T));
    CudaMem src_mem(src.size() * sizeof(T));
    ASSERT_NE(dst_mem.ptr_, nullptr);
    ASSERT_NE(src_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(dst_mem.ptr_, dst0.data(), dst_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(src_mem.ptr_, src.data(), src_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    const std::int64_t n = static_cast<std::int64_t>(host_dst0.size());
    Tensor dst_tensor(dst_mem.ptr_, Traits::dtype, kCuda0, {n});
    Tensor src_tensor(src_mem.ptr_, Traits::dtype, kCuda0, {n});

    ASSERT_TRUE(element_add(&dst_tensor, src_tensor, ExecutionContext{}));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_out(dst0.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), dst_mem.ptr_, dst_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(Traits::to_float(host_out[i]), Traits::to_float(expected[i]), tolerance);
    }
}

std::vector<float> make_values(std::size_t n, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>(i % 7);
    }
    return v;
}

TEST(ElementAddTest, Fp32Basic) {
    const std::vector<float> dst0 = make_values(64, -2.0f, 0.5f);
    const std::vector<float> src = make_values(64, 1.0f, -0.25f);
    run_and_check<float>(dst0, src, 1e-5f);
}

TEST(ElementAddTest, Bf16Basic) {
    const std::vector<float> dst0 = make_values(64, -2.0f, 0.5f);
    const std::vector<float> src = make_values(64, 1.0f, -0.25f);
    run_and_check<__nv_bfloat16>(dst0, src, 1e-2f);
}

TEST(ElementAddTest, SingleElement) { run_and_check<float>({1.5f}, {-0.75f}, 1e-5f); }

TEST(ElementAddTest, NonBlockMultiple) {
    // n = 1000 非 block 整数倍：验证越界守卫。
    const std::vector<float> dst0 = make_values(1000, -1.0f, 0.1f);
    const std::vector<float> src = make_values(1000, 0.5f, -0.05f);
    run_and_check<float>(dst0, src, 1e-5f);
}

TEST(ElementAddTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor dst_tensor(ptr, DType::kFloat32, kCuda0, {4});
    Tensor src_tensor(ptr, DType::kFloat32, kCuda0, {3});
    auto result = element_add(&dst_tensor, src_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(ElementAddTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor dst_tensor(ptr, DType::kFloat16, kCuda0, {4});
    Tensor src_tensor(ptr, DType::kFloat16, kCuda0, {4});
    auto result = element_add(&dst_tensor, src_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
