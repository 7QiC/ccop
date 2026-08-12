#include <cstdint>
#include <string_view>
#include <type_traits>

#include <gtest/gtest.h>

#include "ccop/dtype.h"

namespace ccop {
namespace {

TEST(DTypeTest, TagMapping) {
    EXPECT_EQ(dtype_v<Float32Tag>, DType::kFloat32);
    EXPECT_EQ(dtype_name_v<Float32Tag>, "float32");
    EXPECT_EQ(dtype_size_v<Float32Tag>, sizeof(float));

    EXPECT_EQ(dtype_v<Float16Tag>, DType::kFloat16);
    EXPECT_EQ(dtype_name_v<Float16Tag>, "float16");
    EXPECT_EQ(dtype_size_v<Float16Tag>, 2);

    EXPECT_EQ(dtype_v<BFloat16Tag>, DType::kBFloat16);
    EXPECT_EQ(dtype_name_v<BFloat16Tag>, "bfloat16");
    EXPECT_EQ(dtype_size_v<BFloat16Tag>, 2);

    EXPECT_EQ(dtype_v<Int8Tag>, DType::kInt8);
    EXPECT_EQ(dtype_name_v<Int8Tag>, "int8");
    EXPECT_EQ(dtype_size_v<Int8Tag>, sizeof(std::int8_t));

    EXPECT_EQ(dtype_v<Int32Tag>, DType::kInt32);
    EXPECT_EQ(dtype_name_v<Int32Tag>, "int32");
    EXPECT_EQ(dtype_size_v<Int32Tag>, sizeof(std::int32_t));
}

TEST(DTypeTest, RuntimeLookup) {
    EXPECT_EQ(dtype_size(DType::kBFloat16), 2);
    EXPECT_EQ(dtype_name(DType::kInt8), "int8");
    EXPECT_EQ(dtype_size(DType::kInt32), sizeof(std::int32_t));
    EXPECT_EQ(dtype_name(DType::kInt32), "int32");
    EXPECT_EQ(dtype_size(DType::kUnknown), 0);
    EXPECT_EQ(dtype_name(DType::kUnknown), "");
}

TEST(DTypeTest, NativeTypes) {
    static_assert(std::is_same_v<native_t<Float32Tag>, float>);
    static_assert(std::is_same_v<native_t<Int8Tag>, std::int8_t>);
    static_assert(std::is_same_v<native_t<Int32Tag>, std::int32_t>);
}

}  // namespace
}  // namespace ccop
