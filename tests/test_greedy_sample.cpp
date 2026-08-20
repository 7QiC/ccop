#include "ccop/ops/greedy_sample.h"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "ccop/tensor.h"

namespace ccop {
namespace {

constexpr Device kCuda0{DeviceType::kCUDA, 0};
constexpr std::int32_t kUnchanged = -777;

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

// Host reference：按 indices 逐行 argmax，平局取最小索引；非法行保持不变。
std::vector<std::int32_t> reference_greedy_sample(const std::vector<float>& logits,
                                                  const std::vector<std::int32_t>& indices,
                                                  unsigned int num_tokens,
                                                  unsigned int vocab_size) {
    std::vector<std::int32_t> tokens(indices.size(), kUnchanged);
    for (std::size_t b = 0; b < indices.size(); ++b) {
        const std::int32_t row = indices[b];
        if (row < 0 || row >= static_cast<std::int32_t>(num_tokens)) {
            continue;
        }
        std::int32_t best = 0;
        float best_v = logits[static_cast<std::size_t>(row) * vocab_size];
        for (unsigned int v = 1; v < vocab_size; ++v) {
            const float cur = logits[static_cast<std::size_t>(row) * vocab_size + v];
            if (cur > best_v) {
                best_v = cur;
                best = static_cast<std::int32_t>(v);
            }
        }
        tokens[b] = best;
    }
    return tokens;
}

void run_and_check(const std::vector<float>& logits, const std::vector<std::int32_t>& indices,
                   unsigned int num_tokens, unsigned int vocab_size) {
    ASSERT_EQ(logits.size(), static_cast<std::size_t>(num_tokens) * vocab_size);
    ASSERT_FALSE(indices.empty());
    const std::vector<std::int32_t> expected =
        reference_greedy_sample(logits, indices, num_tokens, vocab_size);

    CudaMem logits_mem(logits.size() * sizeof(float));
    CudaMem indices_mem(indices.size() * sizeof(std::int32_t));
    CudaMem tokens_mem(indices.size() * sizeof(std::int32_t));
    ASSERT_NE(logits_mem.ptr_, nullptr);
    ASSERT_NE(indices_mem.ptr_, nullptr);
    ASSERT_NE(tokens_mem.ptr_, nullptr);
    ASSERT_EQ(cudaMemcpy(logits_mem.ptr_, logits.data(), logits_mem.bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(indices_mem.ptr_, indices.data(), indices_mem.bytes_, cudaMemcpyHostToDevice),
        cudaSuccess);
    std::vector<std::int32_t> tokens_init(indices.size(), kUnchanged);
    ASSERT_EQ(
        cudaMemcpy(tokens_mem.ptr_, tokens_init.data(), tokens_mem.bytes_, cudaMemcpyHostToDevice),
        cudaSuccess);

    Tensor logits_tensor(
        logits_mem.ptr_, DType::kFloat32, kCuda0,
        {static_cast<std::int64_t>(num_tokens), static_cast<std::int64_t>(vocab_size)});
    Tensor indices_tensor(indices_mem.ptr_, DType::kInt32, kCuda0,
                          {static_cast<std::int64_t>(indices.size())});
    Tensor tokens_tensor(tokens_mem.ptr_, DType::kInt32, kCuda0,
                         {static_cast<std::int64_t>(indices.size())});

    ASSERT_TRUE(greedy_sample(logits_tensor, indices_tensor, &tokens_tensor, ExecutionContext{}));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    std::vector<std::int32_t> host_tokens(indices.size());
    ASSERT_EQ(
        cudaMemcpy(host_tokens.data(), tokens_mem.ptr_, tokens_mem.bytes_, cudaMemcpyDeviceToHost),
        cudaSuccess);

    for (std::size_t b = 0; b < expected.size(); ++b) {
        EXPECT_EQ(host_tokens[b], expected[b]);
    }
}

std::vector<float> make_values(std::size_t n, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>(i % 13);
    }
    return v;
}

std::vector<std::int32_t> identity_indices(unsigned int num_tokens) {
    std::vector<std::int32_t> indices(num_tokens);
    for (unsigned int t = 0; t < num_tokens; ++t) {
        indices[t] = static_cast<std::int32_t>(t);
    }
    return indices;
}

TEST(GreedySampleTest, Basic) {
    run_and_check(make_values(3 * 8, -2.0f, 0.5f), identity_indices(3), 3, 8);
}

TEST(GreedySampleTest, SingleToken) {
    run_and_check(make_values(1 * 4, -1.0f, 0.7f), identity_indices(1), 1, 4);
}

TEST(GreedySampleTest, TieArgmax) {
    // 首行两个最大值并列（索引 0 与 2 同为 3.0）：应取最小索引 0。
    const std::vector<float> logits{3.0f, 1.0f, 3.0f, -5.0f, 0.5f, 2.0f, 2.0f, 1.0f};
    run_and_check(logits, identity_indices(2), 2, 4);
}

TEST(GreedySampleTest, AllNegative) {
    // 全负值：最大值是"最不负"的元素。
    const std::vector<float> logits{-5.0f, -1.0f, -3.0f, -8.0f};
    run_and_check(logits, identity_indices(1), 1, 4);
}

TEST(GreedySampleTest, LargeVocab) {
    // vocab = 1000 超 block：验证行内遍历无边界问题。
    run_and_check(make_values(2 * 1000, -3.0f, 0.01f), identity_indices(2), 2, 1000);
}

TEST(GreedySampleTest, IndexedGather) {
    // Prefill/Mixed 语义：logits 行数大于 batch，按 indices 散选。
    run_and_check(make_values(5 * 4, -2.0f, 0.37f), {4, 0, 2}, 5, 4);
}

TEST(GreedySampleTest, SkipInvalidIndices) {
    // -1 与越界行必须跳过，tokens 保持预置值。
    run_and_check(make_values(4 * 4, -1.0f, 0.25f), {-1, 0, 4, 2}, 4, 4);
}

TEST(GreedySampleTest, InvalidArgument) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor logits_tensor(ptr, DType::kFloat32, kCuda0, {2, 4});
    Tensor indices_tensor(ptr, DType::kInt32, kCuda0, {2});
    Tensor tokens_tensor(ptr, DType::kInt32, kCuda0, {3});
    auto result = greedy_sample(logits_tensor, indices_tensor, &tokens_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kInvalidArgument);
}

TEST(GreedySampleTest, UnsupportedDtype) {
    std::vector<float> storage(64);
    void* ptr = storage.data();
    ASSERT_NE(ptr, nullptr);
    Tensor logits_tensor(ptr, DType::kFloat16, kCuda0, {2, 4});
    Tensor indices_tensor(ptr, DType::kInt32, kCuda0, {2});
    Tensor tokens_tensor(ptr, DType::kInt32, kCuda0, {2});
    auto result = greedy_sample(logits_tensor, indices_tensor, &tokens_tensor, ExecutionContext{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::kUnsupported);
}

}  // namespace
}  // namespace ccop
