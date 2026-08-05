#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>

#include "ccop/device.h"
#include "ccop/dtype.h"
#include "ccop/execution_context.h"

namespace ccop {

class Buffer;

inline constexpr int kTensorMaxRank = 8;

struct TensorMeta {
    DType dtype = DType::kUnknown;
    Device device{};
    int rank = 0;
    std::array<std::int64_t, kTensorMaxRank> shape{};
    std::array<std::int64_t, kTensorMaxRank> stride{};
    void* data = nullptr;
};

// 非拥有；算子接口的参数/返回值类型。
class TensorView {
public:
    TensorView() = default;
    TensorView(void* data, DType dtype, Device device,
               std::initializer_list<std::int64_t> shape);
    TensorView(void* data, DType dtype, Device device,
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
    [[nodiscard]] std::int64_t numel() const noexcept;
    [[nodiscard]] std::size_t nbytes() const noexcept;
    [[nodiscard]] bool is_contiguous() const noexcept;
    [[nodiscard]] void* data() noexcept { return meta_.data; }
    [[nodiscard]] const void* data() const noexcept { return meta_.data; }

    [[nodiscard]] TensorView slice(int dim, std::int64_t start,
                                   std::int64_t end) const;
    [[nodiscard]] TensorView select(int dim, std::int64_t index) const;

protected:
    TensorMeta meta_{};
};

// 拥有：RAII over shared_ptr<Buffer>。
class Tensor : public TensorView {
public:
    Tensor() = default;
    Tensor(std::shared_ptr<Buffer> buffer, DType dtype,
           std::initializer_list<std::int64_t> shape);
    Tensor(std::shared_ptr<Buffer> buffer, DType dtype,
           const std::array<std::int64_t, kTensorMaxRank>& shape,
           const std::array<std::int64_t, kTensorMaxRank>& stride, int rank);

    [[nodiscard]] std::shared_ptr<Buffer> buffer() const noexcept { return buffer_; }
    [[nodiscard]] TensorView view() const noexcept;
    [[nodiscard]] Tensor to(Device target, const ExecutionContext& ctx) const;

private:
    std::shared_ptr<Buffer> buffer_;
};

}  // namespace ccop
