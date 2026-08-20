#include "ccop/ops/softmax.h"

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

// Host reference（数值稳定版），输入为 dtype 精确值。
template <typename T>
std::vector<T> reference_softmax(const std::vector<T>& input, unsigned int rows,
                                 unsigned int cols) {
    using Traits = FloatTraits<T>;
    std::vector<T> out(input.size());
    for (unsigned int i = 0; i < rows; ++i) {
        float max_v = Traits::to_float(input[static_cast<std::size_t>(i) * cols]);
        for (unsigned int j = 1; j < cols; ++j) {
            max_v =
                std::max(max_v, Traits::to_float(input[static_cast<std::size_t>(i) * cols + j]));
        }
        float sum = 0.0f;
        for (unsigned int j = 0; j < cols; ++j) {
            sum +=
                std::exp(Traits::to_float(input[static_cast<std::size_t>(i) * cols + j]) - max_v);
        }
        for (unsigned int j = 0; j < cols; ++j) {
            const float v = Traits::to_float(input[static_cast<std::size_t>(i) * cols + j]);
            out[static_cast<std::size_t>(i) * cols + j] =
                Traits::from_float(std::exp(v - max_v) / sum);
        }
    }
    return out;
}

template <typename T>
void run_and_check(const std::vector<float>& host_input, unsigned int rows, unsigned int cols,
                   float tolerance) {
    using Traits = FloatTraits<T>;
    ASSERT_EQ(host_input.size(), static_cast<std::size_t>(rows) * cols);

    std::vector<T> input(host_input.size());
    for (std::size_t i = 0; i < host_input.size(); ++i) {
        input[i] = Traits::from_float(host_input[i]);
    }
    const std::vector<T> expected = reference_softmax(input, rows, cols);

    CudaMem input_mem(input.size() * sizeof(T));
    CudaMem out_mem(input.size() * sizeof(T));
    ASSERT_NE(input_mem.ptr_, nullptr);
    ASSERT_NE(out_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(input_mem.ptr_, input.data(), input_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    Tensor input_tensor(input_mem.ptr_, Traits::dtype, kCuda0,
                        {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)});
    Tensor out_tensor(out_mem.ptr_, Traits::dtype, kCuda0,
                      {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)});

    ASSERT_TRUE(softmax(&out_tensor, input_tensor, ExecutionContext{}));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_out(input.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), out_mem.ptr_, out_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(Traits::to_float(host_out[i]), Traits::to_float(expected[i]), tolerance);
    }
}

std::vector<float> make_input(unsigned int rows, unsigned int cols) {
    std::vector<float> v(static_cast<std::size_t>(rows) * cols);
    for (unsigned int i = 0; i < rows * cols; ++i) {
        v[i] = -1.0f + 0.37f * static_cast<float>(i % 11);
    }
    return v;
}

TEST(SoftmaxTest, Fp32MultiRow) {
    constexpr unsigned int kRows = 4;
    constexpr unsigned int kCols = 8;
    run_and_check<float>(make_input(kRows, kCols), kRows, kCols, 1e-5f);
}

TEST(SoftmaxTest, Bf16MultiRow) {
    constexpr unsigned int kRows = 4;
    constexpr unsigned int kCols = 8;
    run_and_check<__nv_bfloat16>(make_input(kRows, kCols), kRows, kCols, 1e-2f);
}

TEST(SoftmaxTest, SingleRow) {
    constexpr unsigned int kRows = 1;
    constexpr unsigned int kCols = 16;
    run_and_check<float>(make_input(kRows, kCols), kRows, kCols, 1e-5f);
}

TEST(SoftmaxTest, SingleColumn) {
    // cols = 1 时每行 softmax 恒为 1。
    constexpr unsigned int kRows = 3;
    constexpr unsigned int kCols = 1;
    run_and_check<float>(make_input(kRows, kCols), kRows, kCols, 1e-5f);
}

TEST(SoftmaxTest, LargeValues) {
    // 大正数输入：不做 max 减法的实现会 exp 溢出，验证数值稳定性。
    const std::vector<float> input{100.0f, 101.0f, 99.0f, -100.0f};
    run_and_check<float>(input, 1, 4, 1e-5f);
}

TEST(SoftmaxTest, AllNegativeLargeValues) {
    // 全负且低于 exp 下溢阈值：max 必须取行内真实最大值（-110），
    // 否则 exp(-110) 下溢为 0、sum=0 → 0/0=NaN。
    const std::vector<float> input{-110.0f, -120.0f};
    run_and_check<float>(input, 1, 2, 1e-5f);
}

TEST(SoftmaxTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor in_tensor(ptr, DType::kFloat32, kCuda0, {2, 4});
    Tensor out_tensor(ptr, DType::kFloat32, kCuda0, {2, 3});
    auto result = softmax(&out_tensor, in_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(SoftmaxTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor in_tensor(ptr, DType::kFloat16, kCuda0, {2, 4});
    Tensor out_tensor(ptr, DType::kFloat16, kCuda0, {2, 4});
    auto result = softmax(&out_tensor, in_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
