#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "ccop/ops/silu_mul.h"
#include "ccop/tensor.h"

namespace ccop {
namespace {

constexpr Device kCuda0{DeviceType::kCUDA, 0};

struct CudaMem {
    explicit CudaMem(std::size_t bytes) : bytes_(bytes) { cudaMalloc(&ptr_, bytes); }
    ~CudaMem() {
        if (ptr_) cudaFree(ptr_);
    }
    CudaMem(const CudaMem&) = delete;
    CudaMem& operator=(const CudaMem&) = delete;

    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

template <typename T>
struct FloatTraits;

template <>
struct FloatTraits<float> {
    static constexpr DType dtype = DType::kFloat32;
    static float from_float(float v) { return v; }
    static float to_float(float v) { return v; }
};

template <>
struct FloatTraits<__nv_bfloat16> {
    static constexpr DType dtype = DType::kBFloat16;
    static __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
    static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
};

// Host reference: out[i] = silu(gate[i]) * up[i]，输入为 dtype 精确值。
template <typename T>
std::vector<T> reference_silu_mul(const std::vector<T>& gate, const std::vector<T>& up) {
    using Traits = FloatTraits<T>;
    std::vector<T> out(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const float g = Traits::to_float(gate[i]);
        const float silu = g / (1.0f + std::exp(-g));
        out[i] = Traits::from_float(silu * Traits::to_float(up[i]));
    }
    return out;
}

template <typename T>
void run_and_check(const std::vector<float>& host_gate, const std::vector<float>& host_up,
                   float tolerance) {
    using Traits = FloatTraits<T>;
    ASSERT_EQ(host_gate.size(), host_up.size());

    std::vector<T> gate(host_gate.size());
    std::vector<T> up(host_up.size());
    for (std::size_t i = 0; i < host_gate.size(); ++i) {
        gate[i] = Traits::from_float(host_gate[i]);
        up[i] = Traits::from_float(host_up[i]);
    }
    const std::vector<T> expected = reference_silu_mul(gate, up);

    CudaMem gate_mem(gate.size() * sizeof(T));
    CudaMem up_mem(up.size() * sizeof(T));
    CudaMem out_mem(gate.size() * sizeof(T));
    ASSERT_NE(gate_mem.ptr_, nullptr);
    ASSERT_NE(up_mem.ptr_, nullptr);
    ASSERT_NE(out_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(gate_mem.ptr_, gate.data(), gate_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(up_mem.ptr_, up.data(), up_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    const std::int64_t n = static_cast<std::int64_t>(host_gate.size());
    Tensor gate_tensor(gate_mem.ptr_, Traits::dtype, kCuda0, {n});
    Tensor up_tensor(up_mem.ptr_, Traits::dtype, kCuda0, {n});
    Tensor out_tensor(out_mem.ptr_, Traits::dtype, kCuda0, {n});

    silu_mul(&out_tensor, gate_tensor, up_tensor, ExecutionContext{});
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<T> host_out(gate.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), out_mem.ptr_, out_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(Traits::to_float(host_out[i]), Traits::to_float(expected[i]), tolerance);
    }
}

std::vector<float> make_values(std::size_t n, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>(i % 7);
    }
    return v;
}

TEST(SiluMulTest, Fp32MixedSign) {
    const std::vector<float> gate = make_values(64, -3.0f, 0.7f);
    const std::vector<float> up = make_values(64, 0.5f, 0.31f);
    run_and_check<float>(gate, up, 1e-5f);
}

TEST(SiluMulTest, Bf16MixedSign) {
    const std::vector<float> gate = make_values(64, -3.0f, 0.7f);
    const std::vector<float> up = make_values(64, 0.5f, 0.31f);
    run_and_check<__nv_bfloat16>(gate, up, 1e-2f);
}

TEST(SiluMulTest, SingleElement) {
    const std::vector<float> gate{2.0f};
    const std::vector<float> up{-1.5f};
    run_and_check<float>(gate, up, 1e-5f);
}

TEST(SiluMulTest, LargeNegativeGate) {
    // 大绝对值负数 gate：验证 sigmoid 数值稳定（exp(-x) 不溢出）。
    const std::vector<float> gate = make_values(8, -20.0f, 1.0f);
    const std::vector<float> up = make_values(8, 1.0f, 0.25f);
    run_and_check<float>(gate, up, 1e-4f);
}

}  // namespace
}  // namespace ccop
