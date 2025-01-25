#include "gtest/gtest.h"
#include "lattica_hw_api.h"

const NestedVectorType<int32_t, 2> a_cpu = {
    {2147483640, 2147483635, 2147483620},
    {2147483615, 2147483605, 2147483590}
    }; // [2, 3]
const int b_scalar = 2147483625; // Scalar value
const NestedVectorType<int32_t, 1> p_cpu = {
    2147483647, 2147483629, 2147483587
    }; // [3]

TEST(ModOpTestsV3, ModSumV3) {
    auto a_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(a_cpu);
    auto p_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(p_cpu);
    auto result_hw = lattica_hw_api::allocate_on_hardware<int32_t>({2, 3});

    lattica_hw_api::modsum_v3(a_hw, b_scalar, p_hw, result_hw);
    NestedVectorType<int32_t, 2> result_cpu = lattica_hw_api::move_from_hardware<int32_t, 2>(result_hw);

    // Expected values (computed manually)
    NestedVectorType<int32_t, 2> expected_cpu = {{2147483618, 2, 71}, {2147483593, 2147483601, 41}};
    ASSERT_EQ(result_cpu, expected_cpu) << "modsum_v3 failed.";
}

TEST(ModOpTestsV3, ModMulV3) {
    auto a_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(a_cpu);
    auto p_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(p_cpu);
    auto result_hw = lattica_hw_api::allocate_on_hardware<int32_t>({2, 3});

    lattica_hw_api::modmul_v3(a_hw, b_scalar, p_hw, result_hw);
    NestedVectorType<int32_t, 2> result_cpu = lattica_hw_api::move_from_hardware<int32_t, 2>(result_hw);

    // Expected values (computed manually)
    NestedVectorType<int32_t, 2> expected_cpu = {{154, 2147483605, 1254}, {704, 96, 114}};
    ASSERT_EQ(result_cpu, expected_cpu) << "modmul_v3 failed.";
}
