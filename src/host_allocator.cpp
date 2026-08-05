#include "ccop/host_allocator.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>

#include "ccop/buffer.h"
#include "ccop/device.h"

namespace ccop {

namespace {

class HostBuffer final : public Buffer {
public:
    HostBuffer(std::size_t bytes, Device device)
        : bytes_(bytes), device_(device), data_(::operator new(bytes)) {}

    ~HostBuffer() override { ::operator delete(data_); }

    [[nodiscard]] void* data() noexcept override { return data_; }
    [[nodiscard]] const void* data() const noexcept override { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept override { return bytes_; }
    [[nodiscard]] Device device() const noexcept override { return device_; }

private:
    std::size_t bytes_;
    Device device_;
    void* data_;
};

}  // namespace

std::shared_ptr<Buffer> HostAllocator::allocate(std::size_t bytes, Device device) {
    if (!device.is_cpu()) {
        return nullptr;
    }
    return std::make_shared<HostBuffer>(bytes, device);
}

void HostAllocator::copy(void* dst, const void* src, std::size_t bytes,
                         Device dst_device, Device src_device, void* stream) {
    (void)dst_device;
    (void)src_device;
    (void)stream;
    std::memcpy(dst, src, bytes);
}

}  // namespace ccop
