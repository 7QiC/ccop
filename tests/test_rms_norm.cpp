#include "ccop/ops/rms_norm.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
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

// dtype <-> float conversion helpers (host side).
template <typename T>
struct ActivationTraits;

template <>
struct ActivationTraits<float> {
    static constexpr DType dtype = DType::kFloat32;
    static float from_float(float v) { return v; }
    static float to_float(float v) { return v; }
};

template <>
struct ActivationTraits<__nv_bfloat16> {
    static constexpr DType dtype = DType::kBFloat16;
    static __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
    static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
};

// Host reference at float precision; input values are the ones the kernel
// actually sees (already rounded to the activation dtype).
void reference_rms_norm(const float* input, const float* weight, float* output, std::int64_t rows,
                        std::int64_t dim, float eps) {
    for (std::int64_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (std::int64_t d = 0; d < dim; ++d) {
            const float v = input[r * dim + d];
            sum += v * v;
        }
        const float rms = std::sqrt(sum / static_cast<float>(dim) + eps);
        for (std::int64_t d = 0; d < dim; ++d) {
            output[r * dim + d] = input[r * dim + d] / rms * weight[d];
        }
    }
}

template <typename ActivationT, typename WeightT = float>
void run_and_check(const std::vector<float>& host_input, const std::vector<float>& host_weight,
                   std::int64_t rows, std::int64_t dim, float eps, float tolerance) {
    using ActivationTraitsT = ActivationTraits<ActivationT>;
    using WeightTraits = ActivationTraits<WeightT>;
    ASSERT_EQ(host_input.size(), static_cast<std::size_t>(rows * dim));
    ASSERT_EQ(host_weight.size(), static_cast<std::size_t>(dim));

    // Round input and weight to their dtypes, then compute the float reference
    // from the values the kernel actually sees.
    std::vector<ActivationT> in(host_input.size());
    std::vector<float> in_f(host_input.size());
    for (std::size_t i = 0; i < host_input.size(); ++i) {
        in[i] = ActivationTraitsT::from_float(host_input[i]);
        in_f[i] = ActivationTraitsT::to_float(in[i]);
    }
    std::vector<WeightT> w(host_weight.size());
    std::vector<float> w_f(host_weight.size());
    for (std::size_t i = 0; i < host_weight.size(); ++i) {
        w[i] = WeightTraits::from_float(host_weight[i]);
        w_f[i] = WeightTraits::to_float(w[i]);
    }
    std::vector<float> expected_f(host_input.size());
    reference_rms_norm(in_f.data(), w_f.data(), expected_f.data(), rows, dim, eps);

    CudaMem in_mem(in.size() * sizeof(ActivationT));
    CudaMem w_mem(w.size() * sizeof(WeightT));
    CudaMem out_mem(in.size() * sizeof(ActivationT));
    ASSERT_NE(in_mem.ptr_, nullptr);
    ASSERT_NE(w_mem.ptr_, nullptr);
    ASSERT_NE(out_mem.ptr_, nullptr);

    ASSERT_EQ(cudaMemcpy(in_mem.ptr_, in.data(), in_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(w_mem.ptr_, w.data(), w_mem.bytes_, cudaMemcpyHostToDevice), cudaSuccess);

    std::array<std::int64_t, kTensorMaxRank> shape{};
    std::array<std::int64_t, kTensorMaxRank> stride{};
    shape[0] = rows;
    shape[1] = dim;
    stride[0] = dim;
    stride[1] = 1;
    Tensor in_tensor(in_mem.ptr_, ActivationTraitsT::dtype, kCuda0, shape, stride, 2);
    Tensor w_tensor(w_mem.ptr_, WeightTraits::dtype, kCuda0, {dim});
    Tensor out_tensor(out_mem.ptr_, ActivationTraitsT::dtype, kCuda0, shape, stride, 2);

    ASSERT_TRUE(rms_norm(&out_tensor, in_tensor, w_tensor, eps, ExecutionContext{}));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<ActivationT> host_out(in.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), out_mem.ptr_, out_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected_f.size(); ++i) {
        EXPECT_NEAR(ActivationTraitsT::to_float(host_out[i]), expected_f[i], tolerance);
    }
}

TEST(RmsNormTest, Fp32ActivationFp32Weight) {
    const std::vector<float> input{0.1f, 0.2f, 0.3f, 0.4f, 1.0f, -0.5f, 2.0f, 0.7f};
    const std::vector<float> weight{1.0f, 0.5f, 2.0f, 1.5f};
    run_and_check<float>(input, weight, 2, 4, 1e-5f, 1e-5f);
}

TEST(RmsNormTest, Bf16ActivationFp32Weight) {
    // LLM 标准组合：BF16 activation × FP32 weight（混合 dtype 分发）。
    const std::vector<float> input{0.1f, 0.2f, 0.3f, 0.4f, 1.0f, -0.5f, 2.0f, 0.7f};
    const std::vector<float> weight{1.0f, 0.5f, 2.0f, 1.5f};
    run_and_check<__nv_bfloat16>(input, weight, 2, 4, 1e-5f, 1e-2f);
}

TEST(RmsNormTest, Bf16ActivationBf16Weight) {
    // 框架权重按 BF16 保存时的同 dtype 组合。
    const std::vector<float> input{0.1f, 0.2f, 0.3f, 0.4f, 1.0f, -0.5f, 2.0f, 0.7f};
    const std::vector<float> weight{1.0f, 0.5f, 2.0f, 1.5f};
    run_and_check<__nv_bfloat16, __nv_bfloat16>(input, weight, 2, 4, 1e-5f, 1e-2f);
}

TEST(RmsNormTest, SingleRow) {
    const std::vector<float> input{0.5f, -1.5f, 2.0f, 0.25f};
    const std::vector<float> weight{1.0f, 1.0f, 1.0f, 1.0f};
    run_and_check<float>(input, weight, 1, 4, 1e-5f, 1e-5f);
}

TEST(RmsNormTest, DimOne) {
    const std::vector<float> input{3.0f, -2.0f, 0.5f};
    const std::vector<float> weight{0.5f};
    run_and_check<__nv_bfloat16>(input, weight, 3, 1, 1e-5f, 1e-2f);
}

TEST(RmsNormTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor in_tensor(ptr, DType::kFloat32, kCuda0, {2, 4});
    Tensor w_tensor(ptr, DType::kFloat32, kCuda0, {3});
    Tensor out_tensor(ptr, DType::kFloat32, kCuda0, {2, 4});
    auto result = rms_norm(&out_tensor, in_tensor, w_tensor, 1e-5f, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(RmsNormTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor in_tensor(ptr, DType::kFloat16, kCuda0, {1, 4});
    Tensor w_tensor(ptr, DType::kFloat32, kCuda0, {4});
    Tensor out_tensor(ptr, DType::kFloat16, kCuda0, {1, 4});
    auto result = rms_norm(&out_tensor, in_tensor, w_tensor, 1e-5f, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
