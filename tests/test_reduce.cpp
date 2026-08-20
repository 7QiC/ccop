#include "ccop/ops/reduce.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
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

template <typename T>
void run_and_check(const std::vector<float>& input, std::int64_t rows, std::int64_t cols,
                   const std::vector<float>& expected, float tolerance = 1e-4f) {
    using Traits = FloatTraits<T>;
    std::vector<T> in(input.size());
    std::vector<T> expected_t(expected.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        in[i] = Traits::from_float(input[i]);
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected_t[i] = Traits::from_float(expected[i]);
    }

    CudaMem in_mem(in.size() * sizeof(T));
    CudaMem out_mem(expected_t.size() * sizeof(T));
    ASSERT_NE(in_mem.ptr_, nullptr);
    ASSERT_NE(out_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(in_mem.ptr_, in.data(), in_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    std::array<std::int64_t, kTensorMaxRank> shape{};
    std::array<std::int64_t, kTensorMaxRank> stride{};
    shape[0] = rows;
    shape[1] = cols;
    stride[0] = cols;
    stride[1] = 1;
    Tensor in_tensor(in_mem.ptr_, Traits::dtype, kCuda0, shape, stride, 2);
    Tensor out_tensor(out_mem.ptr_, Traits::dtype, kCuda0, {rows});

    ASSERT_TRUE(reduce_sum_rows(&out_tensor, in_tensor, ExecutionContext{}));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_out(expected_t.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), out_mem.ptr_, out_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected_t.size(); ++i) {
        EXPECT_NEAR(Traits::to_float(host_out[i]), Traits::to_float(expected_t[i]), tolerance);
    }
}

TEST(ReduceSumRowsTest, MultipleRows) {
    const std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
    run_and_check<float>(data, 2, 4, {10, 26});
}

TEST(ReduceSumRowsTest, SingleRow) {
    const std::vector<float> data{1, 2, 3, 4, 5};
    run_and_check<float>(data, 1, 5, {15});
}

TEST(ReduceSumRowsTest, OneColumn) {
    const std::vector<float> data{3, 7, 2};
    run_and_check<float>(data, 3, 1, {3, 7, 2});
}

TEST(ReduceSumRowsTest, Bf16Basic) {
    const std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
    run_and_check<__nv_bfloat16>(data, 2, 4, {10, 26}, 1e-2f);
}

TEST(ReduceSumRowsTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor in_tensor(ptr, DType::kFloat32, kCuda0, {4});
    Tensor out_tensor(ptr, DType::kFloat32, kCuda0, {4});
    auto result = reduce_sum_rows(&out_tensor, in_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(ReduceSumRowsTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor in_tensor(ptr, DType::kFloat16, kCuda0, {1, 4});
    Tensor out_tensor(ptr, DType::kFloat16, kCuda0, {1});
    auto result = reduce_sum_rows(&out_tensor, in_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
