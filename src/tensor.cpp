#include "ccop/tensor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "ccop/buffer.h"

namespace ccop {

namespace {

std::int64_t ComputeNumel(int rank, const std::array<std::int64_t, kTensorMaxRank>& shape) {
    std::int64_t result = 1;
    for (int i = 0; i < rank; ++i) {
        result *= shape[i];
    }
    return result;
}

}  // namespace

TensorView::TensorView(void* data, DType dtype, Device device,
                       std::initializer_list<std::int64_t> shape)
    : meta_{dtype, device, static_cast<int>(shape.size()), {}, {}, data} {
    assert(meta_.rank <= kTensorMaxRank);
    std::copy_n(shape.begin(), meta_.rank, meta_.shape.begin());

    if (meta_.rank > 0) {
        meta_.stride[meta_.rank - 1] = 1;
        for (int i = meta_.rank - 2; i >= 0; --i) {
            meta_.stride[i] = meta_.stride[i + 1] * meta_.shape[i + 1];
        }
    }
}

TensorView::TensorView(void* data, DType dtype, Device device,
                       const std::array<std::int64_t, kTensorMaxRank>& shape,
                       const std::array<std::int64_t, kTensorMaxRank>& stride, int rank)
    : meta_{dtype, device, rank, shape, stride, data} {
    assert(0 <= rank && rank <= kTensorMaxRank);
}

std::int64_t TensorView::shape(int dim) const noexcept {
    assert(0 <= dim && dim < meta_.rank);
    return meta_.shape[dim];
}

std::int64_t TensorView::stride(int dim) const noexcept {
    assert(0 <= dim && dim < meta_.rank);
    return meta_.stride[dim];
}

std::int64_t TensorView::numel() const noexcept {
    return ComputeNumel(meta_.rank, meta_.shape);
}

std::size_t TensorView::nbytes() const noexcept {
    return static_cast<std::size_t>(numel()) * dtype_size(meta_.dtype);
}

bool TensorView::is_contiguous() const noexcept {
    std::int64_t expected = 1;
    for (int i = meta_.rank - 1; i >= 0; --i) {
        if (meta_.stride[i] != expected) {
            return false;
        }
        expected *= meta_.shape[i];
    }
    return true;
}

TensorView TensorView::slice(int dim, std::int64_t start, std::int64_t end) const {
    assert(valid());
    assert(0 <= dim && dim < meta_.rank);
    assert(0 <= start && start <= end && end <= meta_.shape[dim]);

    TensorView result = *this;
    result.meta_.shape[dim] = end - start;
    const auto offset = start * result.meta_.stride[dim] *
                        static_cast<std::int64_t>(dtype_size(result.meta_.dtype));
    result.meta_.data = static_cast<std::byte*>(result.meta_.data) + offset;
    return result;
}

TensorView TensorView::select(int dim, std::int64_t index) const {
    assert(valid());
    assert(0 <= dim && dim < meta_.rank);
    assert(0 <= index && index < meta_.shape[dim]);

    TensorView result = *this;
    const auto offset = index * result.meta_.stride[dim] *
                        static_cast<std::int64_t>(dtype_size(result.meta_.dtype));
    result.meta_.data = static_cast<std::byte*>(result.meta_.data) + offset;
    for (int i = dim; i + 1 < result.meta_.rank; ++i) {
        result.meta_.shape[i] = result.meta_.shape[i + 1];
        result.meta_.stride[i] = result.meta_.stride[i + 1];
    }
    --result.meta_.rank;
    return result;
}

Tensor::Tensor(std::shared_ptr<Buffer> buffer, DType dtype,
               std::initializer_list<std::int64_t> shape)
    : TensorView(buffer ? buffer->data() : nullptr, dtype,
                 buffer ? buffer->device() : Device{}, shape),
      buffer_(std::move(buffer)) {}

Tensor::Tensor(std::shared_ptr<Buffer> buffer, DType dtype,
               const std::array<std::int64_t, kTensorMaxRank>& shape,
               const std::array<std::int64_t, kTensorMaxRank>& stride, int rank)
    : TensorView(buffer ? buffer->data() : nullptr, dtype,
                 buffer ? buffer->device() : Device{}, shape, stride, rank),
      buffer_(std::move(buffer)) {}

TensorView Tensor::view() const noexcept {
    return TensorView(meta_.data, dtype(), device(), meta_.shape, meta_.stride, meta_.rank);
}

Tensor Tensor::to(Device target, const ExecutionContext& ctx) const {
    if (target == device()) {
        return Tensor(buffer_, dtype(), meta_.shape, meta_.stride, meta_.rank);
    }
    if (ctx.allocator == nullptr) {
        return Tensor{};
    }

    auto buffer = ctx.allocator->allocate(nbytes(), target);
    if (!buffer) {
        return Tensor{};
    }
    ctx.allocator->copy(buffer->data(), data(), nbytes(), target, device(), ctx.stream);
    return Tensor(std::move(buffer), dtype(), meta_.shape, meta_.stride, meta_.rank);
}

}  // namespace ccop
