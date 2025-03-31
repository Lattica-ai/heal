#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

TEST(GDecompositionEdgeCases, ScalarValues) {
    torch::Tensor a_cpu = torch::tensor({0, 1, 2, 3}, torch::dtype(torch::kInt32));
    int64_t power = 2;
    int64_t base_bits = 1;  // base = 2

    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = allocate_on_hardware<int32_t>({4, power});
    g_decomposition<int32_t>(a_hw, result_hw, power, base_bits);

    torch::Tensor expected = torch::tensor({
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 1}
    }, torch::dtype(torch::kInt32));

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(GDecompositionEdgeCases, ZeroInput) {
    torch::Tensor a_cpu = torch::zeros({5}, torch::dtype(torch::kInt32));
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = allocate_on_hardware<int32_t>({5, 4});
    g_decomposition<int32_t>(a_hw, result_hw, 4, 2);  // base = 4

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::all(result == 0).item<bool>());
}

TEST(GDecompositionEdgeCases, MultiDimensionalInput) {
    torch::Tensor a_cpu = torch::tensor({
        {{5, 12}, {3, 1}},
        {{8, 7}, {9, 2}}
    }, torch::dtype(torch::kInt32));  // [2, 2, 2]
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = allocate_on_hardware<int32_t>({2, 2, 2, 3});  // [2,2,2,3]

    g_decomposition<int32_t>(a_hw, result_hw, 3, 2);  // base = 4

    torch::Tensor expected = torch::tensor({
        {
            { {1, 1, 0}, {0, 3, 0} },
            { {3, 0, 0}, {1, 0, 0} }
        },
        {
            { {0, 2, 0}, {3, 1, 0} },
            { {1, 2, 0}, {2, 0, 0} }
        }
    }, torch::dtype(torch::kInt32));

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(GDecompositionEdgeCases, OverflowWarningCheck) {
    torch::Tensor a_cpu = torch::tensor({255}, torch::dtype(torch::kInt32));
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = allocate_on_hardware<int32_t>({1, 3});

    testing::internal::CaptureStderr();
    g_decomposition<int32_t>(a_hw, result_hw, 3, 3);  // base = 8, max representable = 512
    std::string output = testing::internal::GetCapturedStderr();

    ASSERT_EQ(output.find("exceeds representation capacity"), std::string::npos);
}

TEST(GDecompositionEdgeCases, InvalidShapeMismatch) {
    torch::Tensor a_cpu = torch::tensor({10, 20}, torch::dtype(torch::kInt32));
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = allocate_on_hardware<int32_t>({3, 2});

    EXPECT_THROW(
        g_decomposition<int32_t>(a_hw, result_hw, 2, 2),
        std::invalid_argument
    );
}
