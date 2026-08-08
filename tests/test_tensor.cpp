#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "ccop/tensor.h"

namespace ccop {
namespace {

TEST(TensorTest, ContiguousStrides) {
    std::array<float, 6> storage{};
    Tensor tensor(storage.data(), DType::kFloat32, kCPUDevice, {2, 3});

    EXPECT_EQ(tensor.rank(), 2);
    EXPECT_EQ(tensor.shape(0), 2);
    EXPECT_EQ(tensor.shape(1), 3);
    EXPECT_EQ(tensor.stride(0), 3);
    EXPECT_EQ(tensor.stride(1), 1);
    EXPECT_EQ(tensor.numel(), 6);
    EXPECT_EQ(tensor.nbytes(), 6 * sizeof(float));
    EXPECT_TRUE(tensor.is_contiguous());
    EXPECT_TRUE(tensor.valid());
    EXPECT_FALSE(tensor.is_cuda());
}

TEST(TensorTest, Slice) {
    std::array<float, 6> storage{};
    Tensor tensor(storage.data(), DType::kFloat32, kCPUDevice, {2, 3});

    Tensor sliced = tensor.slice(1, 1, 3);
    EXPECT_EQ(sliced.shape(0), 2);
    EXPECT_EQ(sliced.shape(1), 2);
    EXPECT_EQ(sliced.stride(0), 3);
    EXPECT_EQ(sliced.stride(1), 1);
    EXPECT_EQ(sliced.data(), storage.data() + 1);
    EXPECT_TRUE(sliced.valid());
}

TEST(TensorTest, Select) {
    std::array<float, 6> storage{};
    Tensor tensor(storage.data(), DType::kFloat32, kCPUDevice, {2, 3});

    Tensor selected = tensor.select(1, 2);
    EXPECT_EQ(selected.rank(), 1);
    EXPECT_EQ(selected.shape(0), 2);
    EXPECT_EQ(selected.stride(0), 3);
    EXPECT_EQ(selected.data(), storage.data() + 2);
}

TEST(TensorTest, CopySharesPointerOnly) {
    std::array<float, 6> storage{};
    Tensor tensor(storage.data(), DType::kFloat32, kCPUDevice, {2, 3});

    Tensor copy = tensor;  // pure view: no ownership transfer, no refcount
    EXPECT_EQ(copy.data(), tensor.data());
    EXPECT_EQ(copy.device(), tensor.device());
    EXPECT_EQ(copy.dtype(), tensor.dtype());
    EXPECT_EQ(copy.shape(0), tensor.shape(0));
}

TEST(TensorTest, DefaultIsInvalid) {
    Tensor tensor;
    EXPECT_FALSE(tensor.valid());
    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_EQ(tensor.rank(), 0);
}

}  // namespace
}  // namespace ccop
