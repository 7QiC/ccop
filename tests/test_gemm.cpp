#include "ccop/ops/gemm.h"

#include <cmath>
#include <cstdint>
#include <cublas_v2.h>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

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

struct CublasHandle {
    CublasHandle() { cublasCreate(&handle_); }
    ~CublasHandle() {
        if (handle_ != nullptr) cublasDestroy(handle_);
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t handle_ = nullptr;
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

struct GemmSpec {
    int a_rows = 0;
    int a_cols = 0;
    int b_rows = 0;
    int b_cols = 0;
    bool trans_a = false;
    bool trans_b = false;
    float alpha = 1.0f;
    float beta = 0.0f;
};

int out_m(const GemmSpec& s) { return s.trans_a ? s.a_cols : s.a_rows; }
int out_n(const GemmSpec& s) { return s.trans_b ? s.b_rows : s.b_cols; }
int inner_k(const GemmSpec& s) {
    const int ka = s.trans_a ? s.a_rows : s.a_cols;
    const int kb = s.trans_b ? s.b_cols : s.b_rows;
    return ka == kb ? ka : -1;
}

template <typename T, typename O>
void run_and_check(const std::vector<float>& host_a, const std::vector<float>& host_b,
                   const GemmSpec& spec, float tolerance) {
    using InTraits = FloatTraits<T>;
    using OutTraits = FloatTraits<O>;
    ASSERT_EQ(host_a.size(), static_cast<std::size_t>(spec.a_rows) * spec.a_cols);
    ASSERT_EQ(host_b.size(), static_cast<std::size_t>(spec.b_rows) * spec.b_cols);
    const int m = out_m(spec);
    const int n = out_n(spec);
    const int k = inner_k(spec);
    ASSERT_GT(m, 0);
    ASSERT_GT(n, 0);
    ASSERT_GT(k, 0);

    std::vector<T> a(host_a.size());
    std::vector<T> b(host_b.size());
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = InTraits::from_float(host_a[i]);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = InTraits::from_float(host_b[i]);

    const float c_init_value = 2.0f;
    std::vector<O> c_init(static_cast<std::size_t>(m) * n, OutTraits::from_float(c_init_value));
    std::vector<O> expected(c_init.size());
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc =
                spec.beta * OutTraits::to_float(c_init[static_cast<std::size_t>(i) * n + j]);
            for (int kk = 0; kk < k; ++kk) {
                const float av = InTraits::to_float(a[static_cast<std::size_t>(
                    spec.trans_a ? kk * spec.a_cols + i : i * spec.a_cols + kk)]);
                const float bv = InTraits::to_float(b[static_cast<std::size_t>(
                    spec.trans_b ? j * spec.b_cols + kk : kk * spec.b_cols + j)]);
                acc += spec.alpha * av * bv;
            }
            expected[static_cast<std::size_t>(i) * n + j] = OutTraits::from_float(acc);
        }
    }

    CudaMem a_mem(a.size() * sizeof(T));
    CudaMem b_mem(b.size() * sizeof(T));
    CudaMem c_mem(c_init.size() * sizeof(O));
    ASSERT_NE(a_mem.ptr_, nullptr);
    ASSERT_NE(b_mem.ptr_, nullptr);
    ASSERT_NE(c_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(a_mem.ptr_, a.data(), a_mem.bytes_, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(b_mem.ptr_, b.data(), b_mem.bytes_, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(c_mem.ptr_, c_init.data(), c_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    Tensor a_tensor(a_mem.ptr_, InTraits::dtype, kCuda0, {spec.a_rows, spec.a_cols});
    Tensor b_tensor(b_mem.ptr_, InTraits::dtype, kCuda0, {spec.b_rows, spec.b_cols});
    Tensor c_tensor(c_mem.ptr_, OutTraits::dtype, kCuda0, {m, n});

    CublasHandle handle;
    ASSERT_NE(handle.handle_, nullptr);
    ExecutionContext ctx{nullptr, handle.handle_};

    ASSERT_TRUE(gemm(&c_tensor, a_tensor, b_tensor, spec.trans_a, spec.trans_b, spec.alpha,
                     spec.beta, ctx));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<O> host_out(c_init.size());
    ASSERT_EQ(cudaMemcpy(host_out.data(), c_mem.ptr_, c_mem.bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(OutTraits::to_float(host_out[i]), OutTraits::to_float(expected[i]), tolerance);
    }
}

std::vector<float> make_values(std::size_t n, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>(i % 7);
    }
    return v;
}

TEST(GemmTest, Fp32Basic) {
    run_and_check<float, float>(make_values(2 * 3, -0.5f, 0.25f), make_values(3 * 4, -0.3f, 0.2f),
                                {2, 3, 3, 4}, 1e-5f);
}

TEST(GemmTest, Fp32TransB) {
    run_and_check<float, float>(make_values(2 * 3, -0.5f, 0.25f), make_values(4 * 3, -0.3f, 0.2f),
                                {2, 3, 4, 3, false, true}, 1e-5f);
}

TEST(GemmTest, Bf16ToBf16) {
    run_and_check<__nv_bfloat16, __nv_bfloat16>(
        make_values(2 * 3, -0.5f, 0.25f), make_values(3 * 4, -0.3f, 0.2f), {2, 3, 3, 4}, 1e-2f);
}

TEST(GemmTest, Bf16ToFp32) {
    run_and_check<__nv_bfloat16, float>(make_values(2 * 3, -0.5f, 0.25f),
                                        make_values(3 * 4, -0.3f, 0.2f), {2, 3, 3, 4}, 1e-4f);
}

TEST(GemmTest, BetaAccumulate) {
    run_and_check<float, float>(make_values(2 * 3, -0.5f, 0.25f), make_values(3 * 4, -0.3f, 0.2f),
                                {2, 3, 3, 4, false, false, 1.0f, 0.5f}, 1e-5f);
}

TEST(GemmTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor a_tensor(ptr, DType::kFloat32, kCuda0, {2, 3});
    Tensor b_tensor(ptr, DType::kFloat32, kCuda0, {3, 4});
    Tensor c_tensor(ptr, DType::kFloat32, kCuda0, {2, 5});
    auto result = gemm(&c_tensor, a_tensor, b_tensor, false, false, 1.0f, 0.0f, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(GemmTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor a_tensor(ptr, DType::kFloat16, kCuda0, {2, 3});
    Tensor b_tensor(ptr, DType::kFloat16, kCuda0, {3, 4});
    Tensor c_tensor(ptr, DType::kFloat16, kCuda0, {2, 4});
    auto result = gemm(&c_tensor, a_tensor, b_tensor, false, false, 1.0f, 0.0f, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
