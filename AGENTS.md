# ccop Repository Guidelines

ccop 是一个独立的 C++23 CUDA 算子库，用于练习算子开发全流程：基本实现 → 通用优化 → 面向目标 GPU 定制 → 融合。它以第三方库形式接入 ccInfer 推理框架，替代/演进现有裸指针算子接口。

目标 GPU：NVIDIA GeForce RTX 4060 Laptop（sm_89，24 SM，128-bit 显存带宽，100 KB smem/SM）。未来租用服务器时，按新 GPU 通过 arch/shape 分派定制。

## 目录结构

- `include/ccop/` — 公共头文件：`dtype.h`（tag ↔ native ↔ enum 单一事实源）、`device.h`、`buffer.h`、`execution_context.h`、`tensor.h`、`host_allocator.h`、`cuda/dtype_cuda.h`（仅 CUDA 编译单元包含）
- `src/` — 实现（`tensor.cpp`、`host_allocator.cpp`）
- `tests/` — GTest 单元测试（`test_<module>.cpp`）
- `docs/superpowers/specs/2026-08-06-ccop-interface-design.md` — 接口施工图（骨架阶段的权威接口定义）

## 构建与测试

```bash
cmake -S . -B build -DCCop_BUILD_TESTS=ON
make -C build -j$(nproc)
ctest --test-dir build
```

要求：CMake 3.20+、GCC 13+（C++23）；GTest 在独立构建时通过 FetchContent 自动获取。CUDA kernel（`.cu`）在算子实现阶段加入。

## 架构原则

- **分层**：接口层用 `Tensor`/`TensorView`（host 元数据 + device 指针），kernel 层保持模板 + 裸指针。Tensor 是轻量句柄，严禁引入每步 H2D 同步或"伪 PyTorch"化。
- **Device 与执行上下文分离**：Tensor 携带 `Device`（数据在哪），不携带 Backend/stream（执行上下文）。stream 与 allocator 通过 `ExecutionContext` 由调用方传入。
- **dtype 单一事实源**：tag → native → enum 映射各一份，`static_assert` 保证一致；host 头不依赖 CUDA，f16/bf16 的 native 类型只在 `cuda/dtype_cuda.h` 特化。
- **低耦合**：ccop 不依赖 ccInfer 的任何头文件；ccInfer 通过 FetchContent/submodule 引用 `ccop::ccop`。

## 代码规范

- C++23，命名空间 `ccop`。
- 命名：类/枚举/结构 PascalCase；函数/变量/文件 snake_case；成员尾部下划线（`int count_;`）。
- 头文件 `.h` + `#pragma once`；包含顺序：C 标准 → C++ 标准 → 第三方 → 项目；IWYU，头文件内前向声明，不依赖间接包含。
- 错误处理：骨架阶段用 `assert` + 无效 Tensor（`valid() == false`）表示失败；后续引入错误体系前，热路径不抛异常。
- 所有权：`Buffer` 不可拷贝/移动，由 `std::shared_ptr` 管理；`Tensor` 拷贝共享同一 buffer，`to(同 device)` 零拷贝。

## 测试规范

- GTest，每个模块一个测试文件，命名 `test_<module>.cpp`，`gtest_discover_tests` 注册。
- 正确性：构造 `Tensor` → 调算子接口 → 对拍（naive / PyTorch / vendor），阈值不放松。
- 性能（算子实现阶段）：CUDA event micro-benchmark（warmup + mean/median）+ ncu/nsys，对比 naive / 优化 / vendor / 理论极限。
- 每次改动跑 `ctest` 全量回归；新增算子必须带测试。

## 工程原则

- 不做表面优化：每个优化决策有 profiling 证据支撑。
- 不放松阈值/断言来通过测试；测试失败要修根因。
- 按 GPU 定制：sm_89 优先；新 GPU 通过模板特化 + 运行时分派扩展（CUTLASS 模式），不写死。
- 保持代码干净一致，删除死代码；该要求同样适用于 CMakeLists.txt。

## 与 ccInfer 的对接约定

- ccInfer 通过 FetchContent/submodule 引入 ccop，`using ccinfer::Tensor = ccop::Tensor`（或薄封装）。
- ccInfer 的 Backend 实现 `ccop::Allocator`：`allocate` 封装 cudaMalloc，`copy` 用 `cudaMemcpyAsync(stream)`。
- 算子接口迁移方向：`Params` 裸指针结构体 → `TensorView` 参数 + `ExecutionContext`；每步以 correctness 回归为护栏。
