# ccop

ccop 是一个独立的 C++23 CUDA 算子库，用于练习算子开发全流程：基本实现 → 通用优化 → 面向目标 GPU 定制 → 融合。它以第三方库形式接入 ccInfer 推理框架，替代/演进现有裸指针算子接口。

当前状态：**骨架阶段**。已实现 dtype、Device、Buffer、ExecutionContext、Tensor/TensorView 等基础设施，以及 CPU 端 HostAllocator；CUDA kernel 在算子实现阶段加入。

## 目标 GPU

- NVIDIA GeForce RTX 4060 Laptop（sm_89，24 SM，128-bit 显存带宽，100 KB smem/SM）
- 未来租用服务器时，按新 GPU 通过 arch/shape 分派定制

## 目录结构

```
ccop/
├── CMakeLists.txt
├── README.md
├── include/ccop/
│   ├── dtype.h               // DType enum + tag + name/size/native 单一事实源
│   ├── device.h              // Device{DeviceType, index}
│   ├── buffer.h              // Buffer 抽象（shared_ptr 管理）
│   ├── execution_context.h   // Allocator 抽象 + ExecutionContext
│   ├── tensor.h              // TensorMeta / TensorView / Tensor
│   ├── host_allocator.h      // CPU 端 Allocator 实现
│   └── cuda/dtype_cuda.h     // tag → __half / __nv_bfloat16（仅 CUDA TU）
├── src/
│   ├── tensor.cpp
│   └── host_allocator.cpp
└── tests/
    ├── CMakeLists.txt
    ├── test_dtype.cpp
    ├── test_device.cpp
    └── test_tensor.cpp
```

## 构建与测试

要求：CMake 3.20+、GCC 13+（C++23）。测试开启时通过 FetchContent 自动获取 GTest v1.15.2。

```bash
cmake -S . -B build -DCCop_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

仅构建库（不构建测试）：

```bash
cmake -S . -B build
cmake --build build
```

作为第三方库接入：

```cmake
add_subdirectory(ccop)
target_link_libraries(your_target PRIVATE ccop::ccop)
```

## 使用示例

```cpp
#include <cstdio>

#include "ccop/host_allocator.h"
#include "ccop/tensor.h"

int main() {
    ccop::HostAllocator allocator;
    auto tensor = ccop::Tensor(allocator.allocate(6 * sizeof(float), ccop::kCPUDevice),
                               ccop::DType::kFloat32, {2, 3});
    if (!tensor.valid()) {
        return 1;
    }

    auto* data = static_cast<float*>(tensor.data());
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    ccop::ExecutionContext ctx{nullptr, &allocator};
    auto copy = tensor.to(ccop::Device{ccop::DeviceType::kCPU, 1}, ctx);
    return copy.valid() ? 0 : 1;
}
```

## 设计要点

- **分层**：接口层用 `Tensor`/`TensorView`（host 元数据 + device 指针），kernel 层保持模板 + 裸指针。Tensor 是轻量句柄，不引入每步 H2D 同步。
- **Device 与执行上下文分离**：Tensor 携带 `Device`，不携带 Backend/stream；stream 与 allocator 由调用方通过 `ExecutionContext` 传入。
- **dtype 单一事实源**：tag → native → enum 映射各一份，`static_assert` 保证一致；host 头不依赖 CUDA，f16/bf16 的 native 类型只在 `cuda/dtype_cuda.h` 特化。
- **低耦合**：ccop 不依赖 ccInfer 的任何头文件。
- **错误处理**：骨架阶段用 `assert` + 无效 Tensor（`valid() == false`）表示失败。
- **所有权**：`Buffer` 不可拷贝/移动，由 `std::shared_ptr` 管理；`Tensor` 拷贝共享同一 buffer，`to(同 device)` 零拷贝。

## 与 ccInfer 的对接约定

1. ccInfer 通过 FetchContent/submodule 引入 ccop，`using ccinfer::Tensor = ccop::Tensor`（或薄封装）。
2. Backend（CudaBackend）实现 `ccop::Allocator`：`allocate` 封装 cudaMalloc，`copy` 用 `cudaMemcpyAsync(stream)`。
3. 算子接口从 `Params` 裸指针结构体逐步迁移到 `TensorView` 参数 + `ExecutionContext`。
4. 每步以 correctness 回归（integration tests）为护栏。
