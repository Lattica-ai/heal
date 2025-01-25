#include "gtest/gtest.h"
#include "lattica_hw_api.h"

TEST(ReshapeTests, ReshapeFunctionality) {
    // Define initial vector
    NestedVectorType<int32_t, 3> c_cpu = {{{1, 2, 3}, {4, 5, 6}},
                                 {{7, 8, 9}, {10, 11, 12}}}; // [2, 2, 3]

    auto c_hw = lattica_hw_api::move_to_hardware<int32_t, 3>(c_cpu);

    // Reshape to [6, 2]
    c_hw->reshape({6, 2});
    // Validate content integrity after reshaping
    NestedVectorType<int32_t, 2> expected_after_reshape1 = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}}; // [6, 2]
    NestedVectorType<int32_t, 2> result_after_reshape1 = lattica_hw_api::move_from_hardware<int32_t, 2>(c_hw);
    EXPECT_EQ(result_after_reshape1, expected_after_reshape1) << "Content mismatch after reshape to [6, 2].";


    // Reshape to [3, 4]
    c_hw->reshape({3, 4});
    // Validate content integrity after reshaping
    NestedVectorType<int32_t, 2> expected_after_reshape2 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}; // [3, 4]
    NestedVectorType<int32_t, 2> result_after_reshape2 = lattica_hw_api::move_from_hardware<int32_t, 2>(c_hw);
    EXPECT_EQ(result_after_reshape2, expected_after_reshape2) << "Content mismatch after reshape to [3, 4].";
}
