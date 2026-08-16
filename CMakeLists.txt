cmake_minimum_required(VERSION 3.20)

project(ccop VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

option(CCop_BUILD_TESTS "Build ccop tests" OFF)
include(CTest)

# 供 clangd 等工具消费真实编译参数（.clangd 已指向 build/）。
# 关闭 include 的 response file：clangd 不展开 --options-file 会丢失头文件路径。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Backend implementation selection.
# Public interfaces stay in include/ccop/; backend-specific sources live in
# src/<CCOP_BACKEND>/ (e.g. src/cuda/, future src/cpu/) and are selected here
# at build time.
set(CCOP_BACKEND "cuda" CACHE STRING "Backend implementation to build (cuda)")

find_package(CUDAToolkit REQUIRED)

add_library(ccop src/tensor.cpp)
target_include_directories(ccop PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ccop PUBLIC cxx_std_23)
target_link_libraries(ccop PUBLIC CUDA::cudart)

if(CCOP_BACKEND STREQUAL "cuda")
    target_sources(ccop PRIVATE src/cuda/element_add.cu src/cuda/naive_attention.cu
                                   src/cuda/reduce.cu src/cuda/rms_norm.cu src/cuda/rope.cu
                                   src/cuda/silu_mul.cu src/cuda/softmax.cu
                                   src/cuda/split_qkv.cu src/cuda/write_kv_cache.cu)
endif()

add_library(ccop::ccop ALIAS ccop)

if(CCop_BUILD_TESTS)
    add_subdirectory(tests)
endif()
