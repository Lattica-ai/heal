#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

TEST(MemoryOpsTests, SqueezeRemovesSingletonDim) {
    auto input_cpu = torch::tensor({{{1, 2, 3}}}, torch::kInt64); // shape: [1, 1, 3]
    auto expected_cpu = input_cpu.squeeze(1);                    // shape: [1, 3]

    auto input_hw = host_to_device<int64_t>(input_cpu);
    auto squeezed_hw = squeeze<int64_t>(input_hw, 1);
    auto result_cpu = device_to_host<int64_t>(squeezed_hw);

    ASSERT_TRUE(torch::equal(result_cpu, expected_cpu));
}

TEST(MemoryOpsTests, SqueezeThrowsIfDimNotOne) {
    auto input_cpu = torch::tensor({{1, 2, 3}}, torch::kInt64); // shape: [1, 3]
    auto input_hw = host_to_device<int64_t>(input_cpu);

    EXPECT_THROW(squeeze<int64_t>(input_hw, 1), std::invalid_argument);
}

TEST(MemoryOpsTests, UnsqueezeAddsNewSingletonDim) {
    auto input_cpu = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32); // shape: [2, 2]
    auto expected_cpu = input_cpu.unsqueeze(1);                      // shape: [2, 1, 2]

    auto input_hw = host_to_device<int32_t>(input_cpu);
    auto unsqueezed_hw = unsqueeze<int32_t>(input_hw, 1);
    auto result_cpu = device_to_host<int32_t>(unsqueezed_hw);

    ASSERT_TRUE(torch::equal(result_cpu, expected_cpu));
}

TEST(MemoryOpsTests, UnsqueezeSupportsNegativeAxis) {
    auto input_cpu = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32); // shape: [2, 2]
    auto expected_cpu = input_cpu.unsqueeze(-1);                     // shape: [2, 2, 1]

    auto input_hw = host_to_device<int32_t>(input_cpu);
    auto unsqueezed_hw = unsqueeze<int32_t>(input_hw, -1);
    auto result_cpu = device_to_host<int32_t>(unsqueezed_hw);

    ASSERT_TRUE(torch::equal(result_cpu, expected_cpu));
}

TEST(MemoryOpsTests, UnsqueezeThrowsOnOutOfRangeAxis) {
    auto input_cpu = torch::randint(0, 10, {2, 3}, torch::kInt64); // shape: [2, 3]
    auto input_hw = host_to_device<int64_t>(input_cpu);

    EXPECT_THROW(unsqueeze<int64_t>(input_hw, 4), std::invalid_argument);
    EXPECT_THROW(unsqueeze<int64_t>(input_hw, -4), std::invalid_argument);
}
