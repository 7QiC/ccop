#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "ccop/ops/reduce.h"
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

void run_and_check(const std::vector<float>& input, std::int64_t rows, std::int64_t cols,
                   const std::vector<float>& expected) {
    CudaMem in_mem(input.size() * sizeof(float));
    CudaMem out_mem(expected.size() * sizeof(float));
    ASSERT_NE(in_mem.ptr_, nullptr);
    ASSERT_NE(out_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(in_mem.ptr_, input.data(), input.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    std::array<std::int64_t, kTensorMaxRank> shape{};
    std::array<std::int64_t, kTensorMaxRank> stride{};
    shape[0] = rows;
    shape[1] = cols;
    stride[0] = cols;
    stride[1] = 1;
    Tensor in_tensor(in_mem.ptr_, DType::kFloat32, kCuda0, shape, stride, 2);
    Tensor out_tensor(out_mem.ptr_, DType::kFloat32, kCuda0, {rows});

    reduce_sum_rows(&out_tensor, in_tensor, ExecutionContext{});
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<float> host_out(expected.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), out_mem.ptr_, host_out.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(host_out[i], expected[i], 1e-4);
    }
}

TEST(ReduceSumRowsTest, MultipleRows) {
    const std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
    run_and_check(data, 2, 4, {10, 26});
}

TEST(ReduceSumRowsTest, SingleRow) {
    const std::vector<float> data{1, 2, 3, 4, 5};
    run_and_check(data, 1, 5, {15});
}

TEST(ReduceSumRowsTest, OneColumn) {
    const std::vector<float> data{3, 7, 2};
    run_and_check(data, 3, 1, {3, 7, 2});
}

}  // namespace
}  // namespace ccop
