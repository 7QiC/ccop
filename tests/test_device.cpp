#include <cstdint>

#include <gtest/gtest.h>

#include "ccop/device.h"

namespace ccop {
namespace {

TEST(DeviceTest, Equality) {
    EXPECT_EQ((Device{DeviceType::kCUDA, 0}), (Device{DeviceType::kCUDA, 0}));
    EXPECT_NE((Device{DeviceType::kCUDA, 0}), (Device{DeviceType::kCPU, 0}));
    EXPECT_NE((Device{DeviceType::kCUDA, 0}), (Device{DeviceType::kCUDA, 1}));
}

TEST(DeviceTest, TypePredicates) {
    EXPECT_TRUE(kCPUDevice.is_cpu());
    EXPECT_FALSE(kCPUDevice.is_cuda());
    EXPECT_TRUE((Device{DeviceType::kCUDA, 0}.is_cuda()));
    EXPECT_FALSE((Device{DeviceType::kCUDA, 0}.is_cpu()));
}

TEST(DeviceTest, Layout) {
    static_assert(sizeof(Device) == 8);
}

}  // namespace
}  // namespace ccop
