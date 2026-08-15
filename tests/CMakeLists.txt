include(FetchContent)
include(CTest)
include(GoogleTest)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

foreach(test_name IN ITEMS test_dtype test_device test_tensor test_reduce test_rms_norm test_rope
                        test_silu_mul test_softmax test_element_add test_split_qkv
                        test_write_kv_cache)
    add_executable("${test_name}" "${test_name}.cpp")
    target_link_libraries("${test_name}" PRIVATE ccop::ccop GTest::gtest_main)
    gtest_discover_tests("${test_name}")
endforeach()
