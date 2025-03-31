#include "gtest/gtest.h"
#include "lattica_hw_api.h"

TEST(ReshapeTests, ReshapeFunctionality) {

    // Create initial tensor [2, 2, 3]
    torch::Tensor c_cpu = torch::tensor(
        {{{1, 2, 3}, {4, 5, 6}},
         {{7, 8, 9}, {10, 11, 12}}},
        torch::dtype(torch::kInt32)
    );

    auto c_hw = lattica_hw_api::host_to_device<int32_t>(c_cpu);

    // Reshape to [6, 2]
    c_hw->reshape({6, 2});
    // Validate content integrity after reshaping
    torch::Tensor expected_after_reshape1 = torch::tensor(
        {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}},
        torch::dtype(torch::kInt32)
    ); // [6, 2]
    torch::Tensor result_after_reshape1 = lattica_hw_api::device_to_host<int32_t>(c_hw);
    ASSERT_TRUE(torch::equal(result_after_reshape1, expected_after_reshape1)) << "Content mismatch after reshape to [6, 2].";


    // Reshape to [3, 4]
    c_hw->reshape({3, 4});
    // Validate content integrity after reshaping
    torch::Tensor expected_after_reshape2 = torch::tensor(
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        torch::dtype(torch::kInt32)
    ); // [3, 4]
    torch::Tensor result_after_reshape2 = lattica_hw_api::device_to_host<int32_t>(c_hw);
    ASSERT_TRUE(torch::equal(result_after_reshape2, expected_after_reshape2)) << "Content mismatch after reshape to [3, 4].";
}
