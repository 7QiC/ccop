#include "ccop/ops/rope.h"

#include <cassert>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccop {
namespace {

// Split-half RoPE kernel 声明（接口已定，函数体待实现）。
// 模板参数：T（q/k 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版，不做性能优化；bf162 fast path 留后续）。
//
// 数学形式（逐 pair 二维旋转，split-half 配对）：
//   对每个 pair p，令 i0 = p，i1 = p + half_rotary_dim：
//     [x'[i0]]   [ cos  -sin ] [x[i0]]
//     [x'[i1]] = [ sin   cos ] [x[i1]]
//   即 x'[i0] = x[i0]*cos - x[i1]*sin，x'[i1] = x[i1]*cos + x[i0]*sin。
//   仅前 rotary_dim 维（即 half_rotary_dim 对）参与旋转，其余保持原值。
//
// 数据布局（行主序连续）：
//   q/k: [num_tokens, num_heads, head_dim]，两块独立数组；
//        x[t][h][d] 线性偏移 = (t * num_heads + h) * head_dim + d。
//   positions: [num_tokens] int32，pos = positions[t]。
//   rope_cache: [max_position, half_rotary_dim, 2] float，
//        (pos, p, c) 线性偏移 = (pos * half_rotary_dim + p) * 2 + c，
//        c = 0 为 cos，c = 1 为 sin。
//
// 并行映射：
//   grid.x 覆盖 pair（half_rotary_dim 个；block 内线程数向上取整，需越界守卫）；
//   grid.y 覆盖 token（= num_tokens，精确，无需守卫）；
//   grid.z 覆盖 logical_head（= num_q_heads + num_kv_heads，精确，无需守卫）。
//   logical_head < num_q_heads 时操作 q（head = logical_head），
//   否则操作 k（head = logical_head - num_q_heads）。
//   pos 越界（< 0 或 >= cache_max_position）时不写入。
//
// 数值注意：BF16 参与计算前提升到 float，写回前转回原 dtype；
// 维度与索引统一 unsigned int（uint32 上限约 42.9 亿，覆盖
// 32 头 × 100 万上下文 × head_dim 的目标场景）。
template <typename T>
__global__ void rope_split_half_kernel(T* q, T* k, const int32_t* positions,
                                       const float* rope_cache, unsigned int num_q_heads,
                                       unsigned int num_kv_heads, unsigned int head_dim,
                                       unsigned int half_rotary_dim,
                                       unsigned int cache_max_position) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= half_rotary_dim) {
        return;
    }
    std::int32_t pos = positions[blockIdx.y];
    if (pos < 0 || pos >= cache_max_position) {
        return;
    }
    T* pair1 = nullptr;
    T* pair2 = nullptr;
    if (blockIdx.z < num_q_heads) {
        pair1 = q + (blockIdx.y * num_q_heads + blockIdx.z) * head_dim + tid;
        pair2 = pair1 + half_rotary_dim;
    } else {
        pair1 = k + (blockIdx.y * num_kv_heads + (blockIdx.z - num_q_heads)) * head_dim + tid;
        pair2 = pair1 + half_rotary_dim;
    }
    float cos = rope_cache[(pos * half_rotary_dim + tid) * 2];
    float sin = rope_cache[(pos * half_rotary_dim + tid) * 2 + 1];
    T tmp1 = *pair1;
    T tmp2 = *pair2;
    *pair1 = static_cast<T>(static_cast<float>(tmp1) * cos - static_cast<float>(tmp2) * sin);
    *pair2 = static_cast<T>(static_cast<float>(tmp2) * cos + static_cast<float>(tmp1) * sin);
}

}  // namespace

void rope(Tensor* q, Tensor* k, const Tensor& positions, const Tensor& rope_cache,
          unsigned int rotary_dim, const ExecutionContext& ctx) {
    assert(q != nullptr && q->valid());
    assert(k != nullptr && k->valid());
    assert(q->rank() == 3 && k->rank() == 3);
    assert(q->is_contiguous() && k->is_contiguous());
    assert(q->dtype() == k->dtype());
    assert(q->shape(0) == k->shape(0));
    assert(q->shape(2) == k->shape(2));

    const unsigned int num_tokens = static_cast<unsigned int>(q->shape(0));
    const unsigned int num_q_heads = static_cast<unsigned int>(q->shape(1));
    const unsigned int num_kv_heads = static_cast<unsigned int>(k->shape(1));
    const unsigned int head_dim = static_cast<unsigned int>(q->shape(2));
    assert(rotary_dim > 0 && rotary_dim % 2 == 0 && rotary_dim <= head_dim);

    assert(positions.valid());
    assert(positions.rank() == 1 && positions.shape(0) == num_tokens);
    assert(positions.is_contiguous());
    assert(positions.dtype() == DType::kInt32);

    assert(rope_cache.valid());
    assert(rope_cache.rank() == 3 && rope_cache.is_contiguous());
    assert(rope_cache.dtype() == DType::kFloat32);
    assert(rope_cache.shape(1) == rotary_dim / 2);
    assert(rope_cache.shape(2) == 2);
    const unsigned int cache_max_position = static_cast<unsigned int>(rope_cache.shape(0));
    assert(cache_max_position > 0);

    const unsigned int half_rotary_dim = rotary_dim / 2;
    const unsigned int total_heads = num_q_heads + num_kv_heads;
    const unsigned int block = 64;
    dim3 grid((half_rotary_dim + block - 1) / block, num_tokens, total_heads);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);

    // 组合 dtype 分发：q/k 同 dtype → kernel 模板实例。
    // TODO(operator): 在下面每个 case 中补 launch。
    //   并行映射（与 kernel 注释一致）：
    //     block = 64（scalar 版每线程一个 pair 元素）；
    //     grid.x = ceil(half_rotary_dim / block)，grid.y = num_tokens，
    //     grid.z = num_q_heads + num_kv_heads。
    //   调用要点：
    //     stream 从 ctx.stream 取；q/k/positions/rope_cache 的裸指针从各自
    //     Tensor::data() 解包并转成 kernel 参数类型；模板参数 T 按 dtype 取
    //     __nv_bfloat16（kBFloat16）或 float（kFloat32）；标量按签名顺序传。
    switch (q->dtype()) {
        case DType::kBFloat16:
            rope_split_half_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
                static_cast<__nv_bfloat16*>(q->data()),
                static_cast<__nv_bfloat16*>(k->data()),
                static_cast<const int32_t*>(positions.data()),
                static_cast<const float*>(rope_cache.data()), num_q_heads, num_kv_heads,
                head_dim, half_rotary_dim, cache_max_position);
            break;
        case DType::kFloat32:
            rope_split_half_kernel<float><<<grid, block, 0, s>>>(
                static_cast<float*>(q->data()), static_cast<float*>(k->data()),
                static_cast<const int32_t*>(positions.data()),
                static_cast<const float*>(rope_cache.data()), num_q_heads, num_kv_heads,
                head_dim, half_rotary_dim, cache_max_position);
            break;
        default:
            return;  // 不支持的 q/k dtype
    }
}

}  // namespace ccop
