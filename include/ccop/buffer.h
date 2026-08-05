#pragma once

#include <cstddef>

namespace ccop {

class Device;

// 不可拷贝/移动；由 std::shared_ptr 管理。
class Buffer {
public:
    virtual ~Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    [[nodiscard]] virtual void* data() noexcept = 0;
    [[nodiscard]] virtual const void* data() const noexcept = 0;
    [[nodiscard]] virtual std::size_t bytes() const noexcept = 0;
    [[nodiscard]] virtual Device device() const noexcept = 0;

protected:
    Buffer() = default;
};

}  // namespace ccop
