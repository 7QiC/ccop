#include "ccop/ops/rms_norm.h"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ccop/cuda/error_cuda.h"

namespace ccop {
namespace {

inline bool is_aligned_4(const void* ptr) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0x3u) == 0;
}

inline __device__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

inline __device__ float block_reduce_sum_256(float value) {
    constexpr int kNumWarps = 8;
    __shared__ float warp_sums[kNumWarps];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane < kNumWarps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = block_sum;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <typename ActivationT, typename WeightT>
__global__ void rms_norm_kernel(const ActivationT* input, const WeightT* weight,
                                ActivationT* output, unsigned int rows, unsigned int dim,
                                float eps) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= rows) {
        return;
    }
    float sum = 0.0f;
    for (unsigned int i = 0; i < dim; ++i) {
        float v = static_cast<float>(input[tid * dim + i]);
        sum += v * v;
    }
    float rms = sqrt(sum / dim + eps);
    for (unsigned int i = 0; i < dim; ++i) {
        output[tid * dim + i] = static_cast<ActivationT>(static_cast<float>(input[tid * dim + i]) /
                                                         rms * static_cast<float>(weight[i]));
    }
}

// BF16×BF16 RMSNorm 使用与 HuggingFace/PyTorch 对齐的累加语义：按 bf162
// 成对处理，256 线程跨步，每个 pair term 累加后把部分和舍入回 BF16，
// block reduce 后再舍入一次。
__global__ void rms_norm_bf16_matching_kernel(const __nv_bfloat16* input,
                                              const __nv_bfloat16* weight, __nv_bfloat16* output,
                                              unsigned int rows, unsigned int dim, float eps) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int dim2 = dim / 2;
    const std::uint64_t row_offset = static_cast<std::uint64_t>(row) * dim;

    const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(input + row_offset);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(output + row_offset);

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < dim2; i += 256) {
        float2 x = __bfloat1622float2(in2[i]);
        float term = x.x * x.x + x.y * x.y;
        sum_sq = __bfloat162float(__float2bfloat16_rn(sum_sq + term));
    }
    sum_sq = block_reduce_sum_256(sum_sq);
    sum_sq = __bfloat162float(__float2bfloat16_rn(sum_sq));

    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
    for (unsigned int i = tid; i < dim2; i += 256) {
        float2 x = __bfloat1622float2(in2[i]);
        float2 w = __bfloat1622float2(w2[i]);
        out2[i] = make_bfloat162(__float2bfloat16_rn(x.x * inv_rms * w.x),
                                 __float2bfloat16_rn(x.y * inv_rms * w.y));
    }
}

}  // namespace

Result<void> rms_norm(Tensor* out, const Tensor& input, const Tensor& weight, float eps,
                      const ExecutionContext& ctx) {
    if (!(out != nullptr && out->valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(input.valid() && weight.valid())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(input.rank() == 2 && input.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->rank() == 2 && out->is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->dtype() == input.dtype())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(out->shape(0) == input.shape(0) && out->shape(1) == input.shape(1))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(weight.rank() == 1 && weight.is_contiguous())) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(input.shape(1) == weight.shape(0))) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (!(eps > 0.0f)) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    if (input.shape(0) <= 0 || input.shape(1) <= 0) {
        return std::unexpected(ErrorCode::kInvalidArgument);
    }
    const unsigned int rows = static_cast<unsigned int>(input.shape(0));
    const unsigned int dim = static_cast<unsigned int>(input.shape(1));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    const unsigned int block = 256;
    const unsigned int grid = (rows + block - 1) / block;

    switch (input.dtype()) {
        case DType::kBFloat16: {
            switch (weight.dtype()) {
                case DType::kFloat32:
                    rms_norm_kernel<__nv_bfloat16, float><<<grid, block, 0, s>>>(
                        static_cast<const __nv_bfloat16*>(input.data()),
                        static_cast<const float*>(weight.data()),
                        static_cast<__nv_bfloat16*>(out->data()), rows, dim, eps);
                    return check_cuda_launch();
                case DType::kBFloat16: {
                    const bool can_use_bf162 = (dim % 2 == 0) && is_aligned_4(input.data()) &&
                                               is_aligned_4(weight.data()) &&
                                               is_aligned_4(out->data());
                    if (can_use_bf162) {
                        rms_norm_bf16_matching_kernel<<<rows, block, 0, s>>>(
                            static_cast<const __nv_bfloat16*>(input.data()),
                            static_cast<const __nv_bfloat16*>(weight.data()),
                            static_cast<__nv_bfloat16*>(out->data()), rows, dim, eps);
                    } else {
                        rms_norm_kernel<__nv_bfloat16, __nv_bfloat16><<<grid, block, 0, s>>>(
                            static_cast<const __nv_bfloat16*>(input.data()),
                            static_cast<const __nv_bfloat16*>(weight.data()),
                            static_cast<__nv_bfloat16*>(out->data()), rows, dim, eps);
                    }
                    return check_cuda_launch();
                }
                default:
                    return std::unexpected(ErrorCode::kUnsupported);
            }
        }
        case DType::kFloat32: {
            switch (weight.dtype()) {
                case DType::kFloat32:
                    rms_norm_kernel<float, float>
                        <<<grid, block, 0, s>>>(static_cast<const float*>(input.data()),
                                                static_cast<const float*>(weight.data()),
                                                static_cast<float*>(out->data()), rows, dim, eps);
                    return check_cuda_launch();
                default:
                    return std::unexpected(ErrorCode::kUnsupported);
            }
        }
        default:
            return std::unexpected(ErrorCode::kUnsupported);
    }
}

}  // namespace ccop
