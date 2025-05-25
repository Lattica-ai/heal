#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

TEST(AxisModSumTests, Basic3DAxis1) {
    // Input: [2, 3, 4], reduce over axis=1 → result shape [2, 4]
    torch::Tensor a = torch::tensor({
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}
    }, torch::kInt32);

    torch::Tensor p = torch::tensor({11, 13, 17, 19}, torch::kInt32);
    torch::Tensor expected = (a.sum(1)) % p;

    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({2, 4});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/1);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    std::cout << result << std::endl;
    std::cout << expected << std::endl;
    ASSERT_TRUE(torch::equal(result, expected)) << "3D axis=1 modsum failed.";
}

TEST(AxisModSumTests, ReduceFirstAxis) {
    // Input: [3, 4], reduce over axis=0 → result shape [4]
    torch::Tensor a = torch::tensor({
        {1, 2, 3, 4},
        {4, 5, 6, 7},
        {7, 8, 9, 10}
    }, torch::kInt32);

    torch::Tensor p = torch::tensor({5, 7, 11, 13}, torch::kInt32);
    torch::Tensor expected = (a.sum(0)) % p;

    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({4});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/0);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "2D axis=0 modsum failed.";
}

TEST(AxisModSumTests, HighDimReduction) {
    // Input: [2, 2, 2, 3], reduce over axis=2 → result shape [2, 2, 3]
    torch::Tensor a = torch::arange(24, torch::kInt32).reshape({2, 2, 2, 3});
    torch::Tensor p = torch::tensor({7, 11, 13}, torch::kInt32);
    torch::Tensor expected = (a.sum(2)) % p;

    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({2, 2, 3});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/2);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "High-dim axis=2 modsum failed.";
}

TEST(AxisModSumTests, InvalidAxisThrows) {
    torch::Tensor a = torch::randint(0, 10, {2, 2, 2}, torch::kInt32);
    torch::Tensor p = torch::tensor({7, 11}, torch::kInt32);
    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({2, 2});  // removing axis=2

    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, -1), std::invalid_argument);
    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, 3), std::invalid_argument);
}

TEST(AxisModSumTests, ModulusShapeMismatchThrows) {
    torch::Tensor a = torch::randint(0, 10, {2, 2, 2}, torch::kInt32);
    torch::Tensor p = torch::tensor({7, 11, 13}, torch::kInt32);  // invalid shape
    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({2, 2});

    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, 2), std::invalid_argument);
}
