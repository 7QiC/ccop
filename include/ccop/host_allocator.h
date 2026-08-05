#pragma once

#include <cstddef>
#include <memory>

#include "ccop/buffer.h"
#include "ccop/device.h"
#include "ccop/execution_context.h"

namespace ccop {

class HostAllocator final : public Allocator {
public:
    std::shared_ptr<Buffer> allocate(std::size_t bytes, Device device) override;
    void copy(void* dst, const void* src, std::size_t bytes, Device dst_device,
              Device src_device, void* stream) override;
};

}  // namespace ccop
