#pragma once

#include <expected>

namespace ccop {

enum class ErrorCode {
    kOk,
    kInvalidArgument,
    kUnsupported,
    kOutOfMemory,
    kRuntimeError,
};

template <typename T = void>
using Result = std::expected<T, ErrorCode>;

}  // namespace ccop
