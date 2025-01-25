#include "gtest/gtest.h"
#include "lattica_hw_api.h"

const NestedVectorType<int32_t, 2> a_cpu = {
    {2147483640, 2147483635, 2147483620},
    {2147483615, 2147483605, 2147483590}
    };  // [2, 3]
const NestedVectorType<int32_t, 2> b_cpu = {
    {2147483625, 2147483610, 2147483600},
    {2147483595, 2147483585, 2147483570}
    }; // [2, 3]
const NestedVectorType<int32_t, 1> p_cpu = {
    2147483647, 2147483629, 2147483587
    }; // [3], primes close to 32-bit


TEST(ModOpTestsV1, ModSumV1) {

    auto a_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(a_cpu);
    auto b_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(b_cpu);
    auto p_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(p_cpu);
    auto result_hw = lattica_hw_api::allocate_on_hardware<int32_t>({2, 3});

    lattica_hw_api::modsum_v1(a_hw, b_hw, p_hw, result_hw);
    NestedVectorType<int32_t, 2> result_cpu = lattica_hw_api::move_from_hardware<int32_t, 2>(result_hw);

    // Expected values (computed manually)
    NestedVectorType<int32_t, 2> expected_cpu = {{2147483618, 2147483616, 46}, {2147483563, 2147483561, 2147483573}};
    ASSERT_EQ(result_cpu, expected_cpu);
}

TEST(ModOpTestsV1, ModMulV1) {
    NestedVectorType<int32_t, 2> a_cpu = {{2147483640, 2147483635, 2147483620}, {2147483615, 2147483605, 2147483590}};  // [2, 3]
    NestedVectorType<int32_t, 2> b_cpu = {{2147483625, 2147483610, 2147483600}, {2147483595, 2147483585, 2147483570}}; // [2, 3]
    NestedVectorType<int32_t, 1> p_cpu = {2147483647, 2147483629, 2147483587};                                        // [3], primes close to 32-bit

    auto a_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(a_cpu);
    auto b_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(b_cpu);
    auto p_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(p_cpu);
    auto result_hw = lattica_hw_api::allocate_on_hardware<int32_t>({2, 3});

    lattica_hw_api::modmul_v1(a_hw, b_hw, p_hw, result_hw);
    NestedVectorType<int32_t, 2> result_cpu = lattica_hw_api::move_from_hardware<int32_t, 2>(result_hw);

    // Expected values (computed manually)
    NestedVectorType<int32_t, 2> expected_cpu = {{154, 2147483515, 429}, {1664, 1056, 2147483536}};
    ASSERT_EQ(result_cpu, expected_cpu);
}
