#include "gtest/gtest.h"
#include "lattica_hw_api.h"

TEST(AxisModSumTests, AxisModSum) {
    NestedVectorType<int32_t, 3> c_cpu = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}; // [2, 2, 3]
    NestedVectorType<int32_t, 1> p_cpu = {11, 17, 23};                                         // [3]

    auto c_hw = lattica_hw_api::move_to_hardware<int32_t, 3>(c_cpu);
    auto p_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(p_cpu);
    auto result_hw = lattica_hw_api::allocate_on_hardware<int32_t>({2, 3});

    lattica_hw_api::axis_modsum(c_hw, p_hw, result_hw);
    NestedVectorType<int32_t, 2> result_cpu = lattica_hw_api::move_from_hardware<int32_t, 2>(result_hw);

    // Expected values (computed manually)
    NestedVectorType<int32_t, 2> expected_cpu = {{5, 7, 9}, {6, 2, 21}};
    ASSERT_EQ(result_cpu, expected_cpu);
}
