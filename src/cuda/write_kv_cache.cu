#include "ccop/ops/write_kv_cache.h"

#include <cassert>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccop {
namespace {

// WriteKvCache kernel 声明（接口已定，函数体待实现）。
// 模板参数：T（k_new/v_new/k_cache/v_cache 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版，不做性能优化）。
//
// 数学形式（按 slot 散写，纯搬运无算术）：
//   对每个 token t，slot = slot_mapping[t]：
//     0 <= slot < num_slots 时：
//       k_cache[slot][h][d] = k_new[t][h][d]
//       v_cache[slot][h][d] = v_new[t][h][d]
//     否则跳过（padding/无效 token，不写任何 cache 位置）。
//
// 数据布局（全部行主序连续）：
//   k_new/v_new: [num_tokens, num_kv_heads, head_dim]，
//        线性偏移 = (t * num_kv_heads + h) * head_dim + d；
//   k_cache/v_cache: [num_slots, num_kv_heads, head_dim]，
//        线性偏移 = (slot * num_kv_heads + h) * head_dim + d；
//   slot_mapping: [num_tokens] int32，slot = slot_mapping[t]。
//
// 并行映射：
//   grid.y 覆盖 token（= num_tokens，精确，无需越界守卫）；
//   grid.x 与 block 内 threadIdx.x 共同覆盖 (h, d) 平铺的
//   num_kv_heads * head_dim 个元素（block 向上取整，需越界守卫），
//   元素索引 i = threadIdx.x + blockDim.x * blockIdx.x →
//   h = i / head_dim、d = i % head_dim。
//
// 数值注意：
//   - 纯搬运：同 dtype 直接赋值，无需 float 提升转换；
//   - slot 越界守卫（slot < 0 || slot >= num_slots）在 device 侧做；
//   - 维度与索引统一 unsigned int（uint32 上限约 42.9 亿，覆盖
//     32 头 × 100 万上下文 × head_dim 的目标场景）。
template <typename T>
__global__ void write_kv_cache_kernel(const T* k_new, const T* v_new, T* k_cache, T* v_cache,
                                      const int32_t* slot_mapping, unsigned int num_tokens,
                                      unsigned int num_kv_heads, unsigned int head_dim,
                                      unsigned int num_slots) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int token = blockIdx.y;
    if (tid >= num_kv_heads * head_dim) {
        return;
    }
    int32_t slot = slot_mapping[token];
    if (slot < 0 || slot >= num_slots) {
        return;
    }
    unsigned int h = tid / head_dim;
    unsigned int d = tid % head_dim;
    k_cache[(slot * num_kv_heads + h) * head_dim + d] =
        k_new[(token * num_kv_heads + h) * head_dim + d];
    v_cache[(slot * num_kv_heads + h) * head_dim + d] =
        v_new[(token * num_kv_heads + h) * head_dim + d];
}

}  // namespace

void write_kv_cache(const Tensor& k_new, const Tensor& v_new, Tensor* k_cache, Tensor* v_cache,
                    const Tensor& slot_mapping, const ExecutionContext& ctx) {
    assert(k_new.valid() && v_new.valid());
    assert(k_new.rank() == 3 && k_new.is_contiguous());
    assert(v_new.rank() == 3 && v_new.is_contiguous());
    assert(k_cache != nullptr && k_cache->valid());
    assert(v_cache != nullptr && v_cache->valid());
    assert(k_cache->rank() == 3 && k_cache->is_contiguous());
    assert(v_cache->rank() == 3 && v_cache->is_contiguous());
    assert(k_new.dtype() == v_new.dtype());
    assert(k_new.dtype() == k_cache->dtype());
    assert(k_new.dtype() == v_cache->dtype());

    const unsigned int num_tokens = static_cast<unsigned int>(k_new.shape(0));
    const unsigned int num_kv_heads = static_cast<unsigned int>(k_new.shape(1));
    const unsigned int head_dim = static_cast<unsigned int>(k_new.shape(2));
    assert(num_tokens > 0 && num_kv_heads > 0 && head_dim > 0);
    assert(v_new.shape(0) == static_cast<std::int64_t>(num_tokens));
    assert(v_new.shape(1) == static_cast<std::int64_t>(num_kv_heads));
    assert(v_new.shape(2) == static_cast<std::int64_t>(head_dim));

    const unsigned int num_slots = static_cast<unsigned int>(k_cache->shape(0));
    assert(num_slots > 0);
    assert(k_cache->shape(1) == static_cast<std::int64_t>(num_kv_heads));
    assert(k_cache->shape(2) == static_cast<std::int64_t>(head_dim));
    assert(v_cache->shape(0) == static_cast<std::int64_t>(num_slots));
    assert(v_cache->shape(1) == static_cast<std::int64_t>(num_kv_heads));
    assert(v_cache->shape(2) == static_cast<std::int64_t>(head_dim));

    assert(slot_mapping.valid());
    assert(slot_mapping.rank() == 1 && slot_mapping.is_contiguous());
    assert(slot_mapping.dtype() == DType::kInt32);
    assert(slot_mapping.shape(0) == static_cast<std::int64_t>(num_tokens));

    // 组合 dtype 分发：k/v 四数组同 dtype → kernel 模板实例。
    // TODO(operator): 在下面每个 case 中补 launch。
    //   并行映射（与 kernel 注释一致）：
    //     block = 256（scalar 版每线程一个 (h, d) 元素）；
    //     grid.x = ceil(num_kv_heads * head_dim / block)；
    //     grid.y = num_tokens（精确对应 token，无向上取整）。
    //   调用要点：
    //     stream 从 ctx.stream 取；各数组裸指针从 Tensor::data() 解包并转成
    //     kernel 参数类型（slot_mapping 为 const int32_t*）；模板参数 T 按
    //     dtype 取 __nv_bfloat16（kBFloat16）或 float（kFloat32）；
    //     标量按签名顺序传（num_tokens, num_kv_heads, head_dim, num_slots）。
    const unsigned int block = 256;
    dim3 grid((head_dim * num_kv_heads - 1 + block) / block, num_tokens, 1);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    switch (k_new.dtype()) {
        case DType::kBFloat16:
            write_kv_cache_kernel<__nv_bfloat16>
                <<<grid, block, 0, s>>>(static_cast<const __nv_bfloat16*>(k_new.data()),
                                        static_cast<const __nv_bfloat16*>(v_new.data()),
                                        static_cast<__nv_bfloat16*>(k_cache->data()),
                                        static_cast<__nv_bfloat16*>(v_cache->data()),
                                        static_cast<const int32_t*>(slot_mapping.data()),
                                        num_tokens, num_kv_heads, head_dim, num_slots);
            break;
        case DType::kFloat32:
            write_kv_cache_kernel<float><<<grid, block, 0, s>>>(
                static_cast<const float*>(k_new.data()), static_cast<const float*>(v_new.data()),
                static_cast<float*>(k_cache->data()), static_cast<float*>(v_cache->data()),
                static_cast<const int32_t*>(slot_mapping.data()), num_tokens, num_kv_heads,
                head_dim, num_slots);
            break;
        default:
            return;  // 不支持的 dtype
    }
}

}  // namespace ccop
