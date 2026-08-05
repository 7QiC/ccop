#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "ccop/host_allocator.h"
#include "ccop/tensor.h"

namespace ccop {
namespace {

TEST(TensorViewTest, ContiguousStrides) {
    std::array<float, 6> storage{};
    TensorView view(storage.data(), DType::kFloat32, kCPUDevice, {2, 3});

    EXPECT_EQ(view.rank(), 2);
    EXPECT_EQ(view.shape(0), 2);
    EXPECT_EQ(view.shape(1), 3);
    EXPECT_EQ(view.stride(0), 3);
    EXPECT_EQ(view.stride(1), 1);
    EXPECT_EQ(view.numel(), 6);
    EXPECT_EQ(view.nbytes(), 6 * sizeof(float));
    EXPECT_TRUE(view.is_contiguous());
    EXPECT_TRUE(view.valid());
    EXPECT_TRUE(view.is_cuda() == false);
}

TEST(TensorViewTest, Slice) {
    std::array<float, 6> storage{};
    TensorView view(storage.data(), DType::kFloat32, kCPUDevice, {2, 3});

    TensorView sliced = view.slice(1, 1, 3);
    EXPECT_EQ(sliced.shape(0), 2);
    EXPECT_EQ(sliced.shape(1), 2);
    EXPECT_EQ(sliced.stride(0), 3);
    EXPECT_EQ(sliced.stride(1), 1);
    EXPECT_EQ(sliced.data(), storage.data() + 1);
    EXPECT_TRUE(sliced.valid());
}

TEST(TensorViewTest, Select) {
    std::array<float, 6> storage{};
    TensorView view(storage.data(), DType::kFloat32, kCPUDevice, {2, 3});

    TensorView selected = view.select(1, 2);
    EXPECT_EQ(selected.rank(), 1);
    EXPECT_EQ(selected.shape(0), 2);
    EXPECT_EQ(selected.stride(0), 3);
    EXPECT_EQ(selected.data(), storage.data() + 2);
}

TEST(TensorTest, HostAllocatorRoundTrip) {
    HostAllocator allocator;
    auto tensor = Tensor(allocator.allocate(6 * sizeof(float), kCPUDevice),
                         DType::kFloat32, {2, 3});

    ASSERT_TRUE(tensor.valid());
    EXPECT_EQ(tensor.buffer()->bytes(), 6 * sizeof(float));
    EXPECT_EQ(tensor.buffer()->device(), kCPUDevice);
    EXPECT_EQ(tensor.device(), kCPUDevice);
    EXPECT_EQ(tensor.data(), tensor.buffer()->data());

    auto* data = static_cast<float*>(tensor.data());
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(data[i], static_cast<float>(i));
    }
}

TEST(TensorTest, ToSameDeviceSharesBuffer) {
    HostAllocator allocator;
    ExecutionContext ctx{nullptr, &allocator};
    auto tensor = Tensor(allocator.allocate(6 * sizeof(float), kCPUDevice),
                         DType::kFloat32, {2, 3});

    Tensor same_device = tensor.to(kCPUDevice, ctx);
    EXPECT_TRUE(same_device.valid());
    EXPECT_EQ(same_device.buffer(), tensor.buffer());
}

TEST(TensorTest, ToDifferentCpuIndexCopiesData) {
    HostAllocator allocator;
    ExecutionContext ctx{nullptr, &allocator};
    auto tensor = Tensor(allocator.allocate(6 * sizeof(float), kCPUDevice),
                         DType::kFloat32, {2, 3});
    static_cast<float*>(tensor.data())[2] = 42.0f;

    Tensor copied = tensor.to(Device{DeviceType::kCPU, 1}, ctx);
    ASSERT_TRUE(copied.valid());
    EXPECT_NE(copied.buffer(), tensor.buffer());
    EXPECT_EQ(copied.device(), (Device{DeviceType::kCPU, 1}));
    EXPECT_EQ(static_cast<const float*>(copied.data())[2], 42.0f);
}

TEST(TensorTest, ToWithoutAllocatorIsInvalid) {
    auto tensor = Tensor(std::shared_ptr<Buffer>{}, DType::kFloat32, {2, 3});
    Tensor copied = tensor.to(Device{DeviceType::kCPU, 1}, ExecutionContext{});

    EXPECT_FALSE(copied.valid());
}

}  // namespace
}  // namespace ccop
