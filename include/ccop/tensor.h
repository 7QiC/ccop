#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "ccop/device.h"
#include "ccop/dtype.h"

namespace ccop {

inline constexpr int kTensorMaxRank = 8;

// Pure view: host metadata + raw data pointer. Never owns memory.
//
// Ownership lives in the framework-side Buffer; a Tensor borrows the
// allocation and must not outlive it (std::string_view-style borrowing).
// Ops only read metadata and the data pointer; kernels operate on raw
// pointers, so there is no wrapper overhead on the hot path.
class Tensor {
public:
    Tensor() = default;
    Tensor(void* data, DType dtype, Device device, std::initializer_list<std::int64_t> shape);
    Tensor(void* data, DType dtype, Device device,
           const std::array<std::int64_t, kTensorMaxRank>& shape,
           const std::array<std::int64_t, kTensorMaxRank>& stride, int rank);

    [[nodiscard]] DType dtype() const noexcept { return meta_.dtype; }
    [[nodiscard]] Device device() const noexcept { return meta_.device; }
    [[nodiscard]] int rank() const noexcept { return meta_.rank; }
    [[nodiscard]] bool is_cuda() const noexcept { return meta_.device.is_cuda(); }
    [[nodiscard]] bool valid() const noexcept {
        return meta_.data != nullptr && meta_.rank > 0;
    }
    [[nodiscard]] std::int64_t shape(int dim) const noexcept;
    [[nodiscard]] std::int64_t stride(int dim) const noexcept;
    [[nodiscard]] const std::array<std::int64_t, kTensorMaxRank>& shape() const noexcept {
        return meta_.shape;
    }
    [[nodiscard]] const std::array<std::int64_t, kTensorMaxRank>& stride() const noexcept {
        return meta_.stride;
    }
    [[nodiscard]] std::int64_t numel() const noexcept;
    [[nodiscard]] std::size_t nbytes() const noexcept;
    [[nodiscard]] bool is_contiguous() const noexcept;

    [[nodiscard]] void* data() noexcept { return meta_.data; }
    [[nodiscard]] const void* data() const noexcept { return meta_.data; }

    [[nodiscard]] Tensor slice(int dim, std::int64_t start, std::int64_t end) const;
    [[nodiscard]] Tensor select(int dim, std::int64_t index) const;

private:
    struct Meta {
        DType dtype = DType::kUnknown;
        Device device{};
        int rank = 0;
        std::array<std::int64_t, kTensorMaxRank> shape{};
        std::array<std::int64_t, kTensorMaxRank> stride{};
        void* data = nullptr;
    };

    static void fill_contiguous_strides(Meta& meta);

    Meta meta_{};
};

}  // namespace ccop
