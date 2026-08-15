#include "ccop/ops/split_qkv.h"

#include <cassert>
#include <cmath>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccop {
namespace {

// SplitQkv kernel 声明（接口已定，函数体待实现）。
// 模板参数：T（qkv/q/k/v 的 dtype，BF16 或 FP32）。
// 第一阶段（正确性为先，scalar 版，不做性能优化）。
//
// 数学形式（按输入元素分段搬运，无算术）：
//   每个 token t 的输入行 qkv[t] 长 qkv_dim = (nq + 2*nkv) * hd，
//   前 nq*hd 个元素 → q，接着 nkv*hd 个 → k，最后 nkv*hd 个 → v；
//   各段内部按目标头维度 [num_heads, head_dim] 行主序平铺。
//
// 数据布局（全部行主序连续）：
//   qkv: [num_tokens, qkv_dim]，线性偏移 = t * qkv_dim + off；
//   q:   [num_tokens, nq, hd]，线性偏移 = (t * nq + h) * hd + d；
//   k/v: [num_tokens, nkv, hd]，线性偏移 = (t * nkv + h) * hd + d。
//
// 并行映射（输入驱动）：
//   grid.x 覆盖 qkv 的全部 num_tokens * qkv_dim 个输入元素
//   （block 内线程数向上取整，需越界守卫）；
//   blockIdx.x/threadIdx.x → 输入元素线性索引 idx，
//   由 idx 拆出 token = idx / qkv_dim、off = idx % qkv_dim，
//   off 落在哪个段（< q_dim / < q_dim + k_dim / 其余）决定写到 q/k/v
//   的哪个位置及段内偏移。
//
// 数值注意：
//   - 纯搬运：同 dtype 直接赋值，无需 float 提升转换；
//   - 维度与索引统一 unsigned int（uint32 上限约 42.9 亿，覆盖
//     32 头 × 100 万上下文 × head_dim 的目标场景）。
template <typename T>
__global__ void split_qkv_kernel(const T* qkv, T* q, T* k, T* v, unsigned int num_tokens,
                                 unsigned int num_q_heads, unsigned int num_kv_heads,
                                 unsigned int head_dim) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int qkv_dim = (num_q_heads + num_kv_heads * 2) * head_dim;
    if (tid >= num_tokens * qkv_dim) {
        return;
    }
    unsigned int token = tid / qkv_dim;
    unsigned int offset = tid % qkv_dim;
    T* target = nullptr;
    unsigned int off = 0;
    unsigned int num_head = 0;
    if (offset < head_dim * num_q_heads) {
        target = q;
        num_head = num_q_heads;
        off = offset;
    } else if (offset < head_dim * (num_q_heads + num_kv_heads)) {
        target = k;
        num_head = num_kv_heads;
        off = offset - head_dim * num_q_heads;
    } else {
        target = v;
        num_head = num_kv_heads;
        off = offset - head_dim * (num_q_heads + num_kv_heads);
    }
    unsigned int h = off / head_dim;
    unsigned int d = off % head_dim;
    target[(token * num_head + h) * head_dim + d] = qkv[token * qkv_dim + offset];
}

}  // namespace

void split_qkv(const Tensor& qkv, Tensor* q, Tensor* k, Tensor* v, unsigned int num_q_heads,
               unsigned int num_kv_heads, const ExecutionContext& ctx) {
    assert(qkv.valid());
    assert(qkv.rank() == 2 && qkv.is_contiguous());
    assert(q != nullptr && q->valid());
    assert(k != nullptr && k->valid());
    assert(v != nullptr && v->valid());
    assert(q->rank() == 3 && q->is_contiguous());
    assert(k->rank() == 3 && k->is_contiguous());
    assert(v->rank() == 3 && v->is_contiguous());
    assert(qkv.dtype() == q->dtype());
    assert(qkv.dtype() == k->dtype());
    assert(qkv.dtype() == v->dtype());
    assert(num_q_heads > 0 && num_kv_heads > 0);

    const unsigned int num_tokens = static_cast<unsigned int>(qkv.shape(0));
    const unsigned int head_dim = static_cast<unsigned int>(q->shape(2));
    assert(num_tokens > 0 && head_dim > 0);
    assert(qkv.shape(1) == static_cast<std::int64_t>(num_q_heads + 2 * num_kv_heads) * head_dim);
    assert(q->shape(0) == static_cast<std::int64_t>(num_tokens));
    assert(k->shape(0) == static_cast<std::int64_t>(num_tokens));
    assert(v->shape(0) == static_cast<std::int64_t>(num_tokens));
    assert(q->shape(1) == static_cast<std::int64_t>(num_q_heads));
    assert(k->shape(1) == static_cast<std::int64_t>(num_kv_heads));
    assert(v->shape(1) == static_cast<std::int64_t>(num_kv_heads));
    assert(k->shape(2) == static_cast<std::int64_t>(head_dim));
    assert(v->shape(2) == static_cast<std::int64_t>(head_dim));

    // 组合 dtype 分发：qkv/q/k/v 同 dtype → kernel 模板实例。
    // TODO(operator): 在下面每个 case 中补 launch。
    //   并行映射（与 kernel 注释一致，输入驱动）：
    //     block = 256（scalar 版每线程一个输入元素）；
    //     grid = ceil(num_tokens * qkv_dim / block)，qkv_dim =
    //     (num_q_heads + 2 * num_kv_heads) * head_dim。
    //   调用要点：
    //     stream 从 ctx.stream 取；qkv/q/k/v 的裸指针从各自 Tensor::data()
    //     解包并转成 kernel 参数类型；模板参数 T 按 dtype 取
    //     __nv_bfloat16（kBFloat16）或 float（kFloat32）；标量按签名顺序传。
    const unsigned int qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim;
    const unsigned int block = 256;
    const unsigned int grid = (num_tokens * qkv_dim + block - 1) / block;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream);
    switch (qkv.dtype()) {
        case DType::kBFloat16:
            split_qkv_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
                static_cast<const __nv_bfloat16*>(qkv.data()),
                static_cast<__nv_bfloat16*>(q->data()), static_cast<__nv_bfloat16*>(k->data()),
                static_cast<__nv_bfloat16*>(v->data()), num_tokens, num_q_heads, num_kv_heads,
                head_dim);
            break;
        case DType::kFloat32:
            split_qkv_kernel<float><<<grid, block, 0, s>>>(
                static_cast<const float*>(qkv.data()), static_cast<float*>(q->data()),
                static_cast<float*>(k->data()), static_cast<float*>(v->data()), num_tokens,
                num_q_heads, num_kv_heads, head_dim);
            break;
        default:
            return;  // 不支持的 dtype
    }
}

}  // namespace ccop
