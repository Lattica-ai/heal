#include "gtest/gtest.h"
#include "lattica_hw_api.h"

TEST(PermuteTests, ApplyPermutations) {
    NestedVectorType<int32_t, 3> a_cpu = {{{1}, {2}, {3}, {4}},
                                 {{5}, {6}, {7}, {8}},
                                 {{9}, {10}, {11}, {12}}}; // [3, 4, 1]
    NestedVectorType<int32_t, 2> perms_cpu = {{2, 0, 3, 1},
                                     {3, 2, 1, 0},
                                     {0, 3, 1, 2}};          // [3, 4]

    auto a_hw = lattica_hw_api::move_to_hardware<int32_t, 3>(a_cpu);
    auto perms_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(perms_cpu);
    auto result_hw = lattica_hw_api::allocate_on_hardware<int32_t>({3, 4, 1});

    lattica_hw_api::permute<int32_t>(a_hw, perms_hw, result_hw);\

    NestedVectorType<int32_t, 3> result_cpu = lattica_hw_api::move_from_hardware<int32_t, 3>(result_hw);

    NestedVectorType<int32_t, 3> expected_cpu = {{{3}, {1}, {4}, {2}},
                                        {{8}, {7}, {6}, {5}},
                                        {{9}, {12}, {10}, {11}}};
    ASSERT_EQ(result_cpu, expected_cpu) << "Permutations were not applied correctly.";
}
