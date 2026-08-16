#include "ccop/ops/greedy_sample.h"

#include <cstdint>
#include <vector>

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

// Host reference：逐行 argmax，平局取最小索引。
std::vector<std::int32_t> reference_greedy_sample(const std::vector<float>& logits,
                                                  unsigned int num_tokens,
                                                  unsigned int vocab_size) {
    std::vector<std::int32_t> tokens(num_tokens);
    for (unsigned int t = 0; t < num_tokens; ++t) {
        std::int32_t best = 0;
        float best_v = logits[static_cast<std::size_t>(t) * vocab_size];
        for (unsigned int v = 1; v < vocab_size; ++v) {
            const float cur = logits[static_cast<std::size_t>(t) * vocab_size + v];
            if (cur > best_v) {
                best_v = cur;
                best = static_cast<std::int32_t>(v);
            }
        }
        tokens[t] = best;
    }
    return tokens;
}

void run_and_check(const std::vector<float>& logits, unsigned int num_tokens,
                   unsigned int vocab_size) {
    ASSERT_EQ(logits.size(), static_cast<std::size_t>(num_tokens) * vocab_size);
    const std::vector<std::int32_t> expected =
        reference_greedy_sample(logits, num_tokens, vocab_size);

    CudaMem logits_mem(logits.size() * sizeof(float));
    CudaMem tokens_mem(num_tokens * sizeof(std::int32_t));
    ASSERT_NE(logits_mem.ptr_, nullptr);
    ASSERT_NE(tokens_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(logits_mem.ptr_, logits.data(), logits_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);

    Tensor logits_tensor(
        logits_mem.ptr_, DType::kFloat32, kCuda0,
        {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(vocab_size)});
    Tensor tokens_tensor(tokens_mem.ptr_, DType::kInt32, kCuda0,
                         {static_cast<std::int64_t>(num_tokens)});

    greedy_sample(logits_tensor, &tokens_tensor, ExecutionContext{});
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<std::int32_t> host_tokens(num_tokens);
    ASSERT_EQ(
        cudaMemcpy(host_tokens.data(), tokens_mem.ptr_, tokens_mem.bytes_, cudaMemcpyDeviceToHost),
        cudaSuccess);

    for (std::size_t t = 0; t < expected.size(); ++t) {
        EXPECT_EQ(host_tokens[t], expected[t]);
    }
}

std::vector<float> make_values(std::size_t n, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>(i % 13);
    }
    return v;
}

TEST(GreedySampleTest, Basic) { run_and_check(make_values(3 * 8, -2.0f, 0.5f), 3, 8); }

TEST(GreedySampleTest, SingleToken) { run_and_check(make_values(1 * 4, -1.0f, 0.7f), 1, 4); }

TEST(GreedySampleTest, TieArgmax) {
    // 首行两个最大值并列（索引 0 与 2 同为 3.0）：应取最小索引 0。
    const std::vector<float> logits{3.0f, 1.0f, 3.0f, -5.0f, 0.5f, 2.0f, 2.0f, 1.0f};
    run_and_check(logits, 2, 4);
}

TEST(GreedySampleTest, AllNegative) {
    // 全负值：最大值是"最不负"的元素。
    const std::vector<float> logits{-5.0f, -1.0f, -3.0f, -8.0f};
    run_and_check(logits, 1, 4);
}

TEST(GreedySampleTest, LargeVocab) {
    // vocab = 1000 超 block：验证行内遍历无边界问题。
    run_and_check(make_values(2 * 1000, -3.0f, 0.01f), 2, 1000);
}

}  // namespace
}  // namespace ccop
