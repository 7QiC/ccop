#pragma once

#include <cstddef>
#include <memory>

namespace ccop {

class Buffer;
struct Device;

class Allocator {
public:
    virtual ~Allocator() = default;

    virtual std::shared_ptr<Buffer> allocate(std::size_t bytes, Device device) = 0;
    virtual void copy(void* dst, const void* src, std::size_t bytes, Device dst_device,
                      Device src_device, void* stream) = 0;

protected:
    Allocator() = default;
};

struct ExecutionContext {
    void* stream = nullptr;  // 如 cudaStream_t；null = 默认流
    Allocator* allocator = nullptr;
};

}  // namespace ccop
