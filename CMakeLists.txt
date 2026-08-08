cmake_minimum_required(VERSION 3.20)

project(ccop VERSION 0.1.0 LANGUAGES CXX)

option(CCop_BUILD_TESTS "Build ccop tests" OFF)
include(CTest)

# Backend implementation selection.
# Public interfaces stay in include/ccop/; backend-specific sources live in
# src/<CCOP_BACKEND>/ (e.g. src/cuda/, future src/cpu/) and are selected here
# at build time.
set(CCOP_BACKEND "cuda" CACHE STRING "Backend implementation to build (cuda)")

add_library(
    ccop
    src/tensor.cpp
)
target_include_directories(ccop PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ccop PUBLIC cxx_std_23)
add_library(ccop::ccop ALIAS ccop)

if(CCop_BUILD_TESTS)
    add_subdirectory(tests)
endif()
