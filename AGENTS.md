# ccop Repository Guidelines

ccop 是一个独立的 C++23 CUDA 算子库，用于练习算子开发全流程：基本实现 → 通用优化 → 面向目标 GPU 定制 → 融合。它以第三方库形式接入 ccInfer 推理框架，替代/演进现有裸指针算子接口。

目标 GPU：NVIDIA GeForce RTX 4060 Laptop（sm_89，24 SM，128-bit 显存带宽，100 KB smem/SM）。未来租用服务器时，按新 GPU 通过 arch/shape 分派定制。

## 目录结构

- `include/ccop/` — 公共接口：`dtype.h`（tag ↔ native ↔ enum 单一事实源）、`device.h`、`execution_context.h`（stream + 可选 blas handle）、`tensor.h`（纯视图）、`error.h`、`cuda/dtype_cuda.h` / `cuda/error_cuda.h`（仅 CUDA 编译单元包含）
- `src/` — 通用实现（`tensor.cpp`）；后端特定实现放在 `src/<backend>/`（如 `src/cuda/`），由 `CCOP_BACKEND` 编译选项选择
- `tests/` — GTest 单元测试（`test_<module>.cpp`），测试自带存储/allocator，不进入公共 API
- `docs/superpowers/specs/2026-08-06-ccop-interface-design.md` — 接口施工图（历史版本，以实际代码为准）

## 构建与测试

```bash
cmake -S . -B build -DCCop_BUILD_TESTS=ON
make -C build -j$(nproc)
ctest --test-dir build
```

要求：CMake 3.20+、GCC 13+（C++23）；GTest 在独立构建时通过 FetchContent 自动获取。CUDA kernel（`.cu`）在算子实现阶段加入，放在 `src/cuda/`。

## 架构原则

- **Tensor 是纯视图**：host 元数据 + 裸 `data_ptr`，不拥有内存、无引用计数。显存所有权归框架侧 `Buffer`；Tensor 借用底层分配，不得活得比它久（`std::string_view` 风格）。没有 TensorView——Tensor 本身就是视图。
- **算子库只管算**：不负责显存分配、不持有长期资源、不包含 allocator。kernel 直接操作裸指针 + 模板，零包装开销。GEMM 当前封装 cuBLAS（handle 由调用方通过 ExecutionContext 提供），后续再替换为自研 kernel。
- **Device 与执行上下文分离**：Tensor 携带 `Device`（数据在哪）；`ExecutionContext` 含不透明 `stream` 与可选 `blas_handle`（均由调用方传入，ccop 不拥有）。
- **dtype 单一事实源**：tag → native → enum 映射各一份，`static_assert` 保证一致；host 头不依赖 CUDA，f16/bf16 的 native 类型只在 `cuda/dtype_cuda.h` 特化。
- **低耦合**：ccop 不依赖 ccInfer 的任何头文件；ccInfer 通过 submodule 引用 `ccop::ccop`。

## 算子开发协作模式（强制要求）

本项目以提升用户算子能力为目标，Agent（Codex）与用户的协作方式固定如下：

1. **Agent 只搭框架，不写算子实现内容**：函数签名、分派骨架、`// TODO(operator): implement ...` 注释。算子核心逻辑（kernel、tiling、softmax、融合等）由用户自己写。
   - **引导注释规范（强制）**：TODO 引导只描述问题的数学形式（公式/矩阵）、数据布局（形状、线性偏移）、并行映射（grid/block/thread 的 x/y/z 语义、边界守卫的来源）与数值注意点；**不得给出逐行代码步骤或可照抄的实现片段**。用户根据这些信息自行推导实现，以实现练习为第一目的。
2. **Agent 写测试用例**：每个新算子必须带 GTest 测试（正确性对拍 naive/PyTorch/vendor、边界 shape、dtype 检查）。
3. **用户补充实现后，由 Agent 跑测试、review 代码、指出错误**：Agent 给出具体错误位置和原因，用户继续修改，循环直到全绿。
4. **算子优化同样遵守**：Agent 提供 profiling 证据和优化方向，不替用户写优化后的 kernel；用户实现后 Agent 复测。
5. Agent 不得为了通过测试而替用户改写算子核心实现；只允许修正接口、测试基建、编译/链接问题。

## 代码规范

- C++23，命名空间 `ccop`。
- 命名：类/枚举/结构 PascalCase；函数/变量/文件 snake_case；成员尾部下划线（`int count_;`）。
- 参数语义：**只读参数用 `const T&`；会被修改的参数不用引用，直接传指针 `T*`**（输出/原地修改参数必须是指针，调用处 `&x` 显式可见）。
- 数值类型约定：维度、网格尺寸（block/grid）、kernel 索引（tid 等）统一用 `unsigned int`（32 位足够：uint32 上限约 42.9 亿，覆盖 32 头 × 100 万上下文 × head_dim 的目标场景）；不引入 64 位索引。
- 头文件 `.h` + `#pragma once`；包含顺序：源文件对应头（主头，由 `.clang-format` 的 IncludeIsMainRegex 识别）第一 → C/C++ 标准库 → 第三方（CUDA/gtest）→ 其他项目头，组内按字母序；IWYU，头文件内前向声明，不依赖间接包含。
- 错误处理：骨架阶段用 `assert` + 无效 Tensor（`valid() == false`）表示失败；后续引入错误体系前，热路径不抛异常。
- 内存：显存所有权在框架侧，ccop 不分配/不释放用户内存；测试内部可有自己的 test allocator。
- 注释：不写多余注释，只在必要处写简洁注释；不逐行复述代码，不把对话中的决策写进代码注释。kernel 函数侧例外——按算子实现指导保留所需的提示/引导注释（如 `// TODO(operator): implement ...`、tiling/优化方向提示），这类注释属于协作引导内容，不得删减。

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

- ccInfer 通过 submodule 引入 ccop（`ccop::ccop`），公共接口统一从 `include/ccop/` 使用。
- 显存所有权：ccInfer 的 `backend/buffer.h` 的 `Buffer` 持有内存；算子入口接收 `ccop::Tensor`（纯视图）+ `ccop::ExecutionContext`。
- 算子接口迁移方向：ccInfer 现有 `Params` 裸指针结构体 → `ccop::Tensor` 参数 + `ExecutionContext`；每步以 correctness 回归为护栏。
