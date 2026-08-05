#pragma once

#include <cstdint>

namespace ccop {

enum class DeviceType : std::uint8_t { kCPU = 0, kCUDA };

struct Device {
    DeviceType type = DeviceType::kCPU;
    std::int32_t index = 0;

    [[nodiscard]] constexpr bool is_cpu() const noexcept {
        return type == DeviceType::kCPU;
    }
    [[nodiscard]] constexpr bool is_cuda() const noexcept {
        return type == DeviceType::kCUDA;
    }
    [[nodiscard]] constexpr bool operator==(const Device&) const noexcept = default;
};

inline constexpr Device kCPUDevice{DeviceType::kCPU, 0};

static_assert(sizeof(Device) == 8);

}  // namespace ccop
