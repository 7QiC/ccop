#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ccop {

enum class DType : std::uint8_t {
    kUnknown = 0,
    kFloat32,
    kFloat16,
    kBFloat16,
    kInt8,
};

struct Float32Tag {};
struct Float16Tag {};
struct BFloat16Tag {};
struct Int8Tag {};

// 单一事实源：tag -> enum / name / byte size。
template <typename Tag>
struct DTypeTraits;

template <>
struct DTypeTraits<Float32Tag> {
    static constexpr DType dtype = DType::kFloat32;
    static constexpr std::string_view name = "float32";
    static constexpr std::size_t size = sizeof(float);
};

template <>
struct DTypeTraits<Float16Tag> {
    static constexpr DType dtype = DType::kFloat16;
    static constexpr std::string_view name = "float16";
    static constexpr std::size_t size = 2;
};

template <>
struct DTypeTraits<BFloat16Tag> {
    static constexpr DType dtype = DType::kBFloat16;
    static constexpr std::string_view name = "bfloat16";
    static constexpr std::size_t size = 2;
};

template <>
struct DTypeTraits<Int8Tag> {
    static constexpr DType dtype = DType::kInt8;
    static constexpr std::string_view name = "int8";
    static constexpr std::size_t size = sizeof(std::int8_t);
};

template <typename Tag>
inline constexpr DType dtype_v = DTypeTraits<Tag>::dtype;

template <typename Tag>
inline constexpr std::string_view dtype_name_v = DTypeTraits<Tag>::name;

template <typename Tag>
inline constexpr std::size_t dtype_size_v = DTypeTraits<Tag>::size;

inline constexpr std::size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::kFloat32:
            return dtype_size_v<Float32Tag>;
        case DType::kFloat16:
            return dtype_size_v<Float16Tag>;
        case DType::kBFloat16:
            return dtype_size_v<BFloat16Tag>;
        case DType::kInt8:
            return dtype_size_v<Int8Tag>;
        case DType::kUnknown:
            return 0;
    }
    return 0;
}

inline constexpr std::string_view dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::kFloat32:
            return dtype_name_v<Float32Tag>;
        case DType::kFloat16:
            return dtype_name_v<Float16Tag>;
        case DType::kBFloat16:
            return dtype_name_v<BFloat16Tag>;
        case DType::kInt8:
            return dtype_name_v<Int8Tag>;
        case DType::kUnknown:
            return {};
    }
    return {};
}

// host 可表示类型：float / int8_t；f16/bf16 仅在 cuda/dtype_cuda.h 特化。
template <typename Tag>
struct NativeOf;

template <>
struct NativeOf<Float32Tag> {
    using type = float;
};

template <>
struct NativeOf<Int8Tag> {
    using type = std::int8_t;
};

template <typename Tag>
using native_t = typename NativeOf<Tag>::type;

static_assert(dtype_size_v<Float32Tag> == sizeof(native_t<Float32Tag>));
static_assert(dtype_size_v<Int8Tag> == sizeof(native_t<Int8Tag>));

}  // namespace ccop
