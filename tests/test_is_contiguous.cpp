#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

TEST(IsContiguousTests, TensorNotContiguous) {
    auto t = torch::arange(12, torch::kInt32).reshape({3, 4}).transpose(0, 1); // not contiguous
    auto t_hw = host_to_device<int32_t>(t);
    ASSERT_FALSE(t_hw->is_contiguous());
}

TEST(IsContiguousTests, TensorContiguous) {
    auto t = torch::arange(12, torch::kInt32); // contiguous
    auto t_hw = host_to_device<int32_t>(t);
    ASSERT_TRUE(t_hw->is_contiguous());
}