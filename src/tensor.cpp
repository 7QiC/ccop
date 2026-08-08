#include "ccop/tensor.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ccop {

void Tensor::fill_contiguous_strides(Tensor::Meta& meta) {
    std::int64_t stride = 1;
    for (int d = meta.rank - 1; d >= 0; --d) {
        meta.stride[d] = stride;
        stride *= meta.shape[d];
    }
}

Tensor::Tensor(void* data, DType dtype, Device device,
               std::initializer_list<std::int64_t> shape) {
    assert(static_cast<int>(shape.size()) <= kTensorMaxRank);
    meta_.dtype = dtype;
    meta_.device = device;
    meta_.data = data;
    meta_.rank = static_cast<int>(shape.size());
    int i = 0;
    for (auto s : shape) {
        meta_.shape[i++] = s;
    }
    fill_contiguous_strides(meta_);
}

Tensor::Tensor(void* data, DType dtype, Device device,
               const std::array<std::int64_t, kTensorMaxRank>& shape,
               const std::array<std::int64_t, kTensorMaxRank>& stride, int rank) {
    assert(rank >= 0 && rank <= kTensorMaxRank);
    meta_.dtype = dtype;
    meta_.device = device;
    meta_.data = data;
    meta_.rank = rank;
    meta_.shape = shape;
    meta_.stride = stride;
}

std::int64_t Tensor::shape(int dim) const noexcept {
    assert(dim >= 0 && dim < meta_.rank);
    return meta_.shape[dim];
}

std::int64_t Tensor::stride(int dim) const noexcept {
    assert(dim >= 0 && dim < meta_.rank);
    return meta_.stride[dim];
}

std::int64_t Tensor::numel() const noexcept {
    std::int64_t n = 1;
    for (int d = 0; d < meta_.rank; ++d) {
        n *= meta_.shape[d];
    }
    return n;
}

std::size_t Tensor::nbytes() const noexcept {
    return static_cast<std::size_t>(numel()) * dtype_size(meta_.dtype);
}

bool Tensor::is_contiguous() const noexcept {
    std::int64_t expected = 1;
    for (int d = meta_.rank - 1; d >= 0; --d) {
        if (meta_.stride[d] != expected) {
            return false;
        }
        expected *= meta_.shape[d];
    }
    return true;
}

Tensor Tensor::slice(int dim, std::int64_t start, std::int64_t end) const {
    assert(dim >= 0 && dim < meta_.rank);
    assert(start >= 0 && end >= start && end <= meta_.shape[dim]);

    Tensor t = *this;
    t.meta_.shape[dim] = end - start;
    t.meta_.data = static_cast<char*>(t.meta_.data) +
                   start * meta_.stride[dim] * dtype_size(meta_.dtype);
    return t;
}

Tensor Tensor::select(int dim, std::int64_t index) const {
    assert(dim >= 0 && dim < meta_.rank);
    assert(index >= 0 && index < meta_.shape[dim]);

    Tensor t = *this;
    t.meta_.data = static_cast<char*>(t.meta_.data) +
                   index * meta_.stride[dim] * dtype_size(meta_.dtype);
    for (int i = dim; i < meta_.rank - 1; ++i) {
        t.meta_.shape[i] = meta_.shape[i + 1];
        t.meta_.stride[i] = meta_.stride[i + 1];
    }
    --t.meta_.rank;
    t.meta_.shape[t.meta_.rank] = 0;
    t.meta_.stride[t.meta_.rank] = 0;
    return t;
}

}  // namespace ccop
