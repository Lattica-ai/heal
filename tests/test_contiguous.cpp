#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

TEST(ContiguousTests, MakesTensorContiguous) {
    auto t = torch::arange(12, torch::kInt32).reshape({3, 4}).transpose(0, 1);  // not contiguous
    auto hw = host_to_device<int32_t>(t);
    auto hw_contig = contiguous<int32_t>(hw);
    auto back = device_to_host<int32_t>(hw_contig);
    ASSERT_TRUE(torch::equal(back, t));
}

TEST(ContiguousTests, ReturnsSameIfAlreadyContiguous) {
    auto t = torch::randint(0, 60000, {5, 6}, torch::kInt64);
    auto hw = host_to_device<int64_t>(t);
    auto result = contiguous<int64_t>(hw);
    ASSERT_EQ(hw.get(), result.get());  // Same pointer
}

TEST(ContiguousTests, IsContiguousTensorNotContiguous) {
    auto t = torch::arange(12, torch::kInt32).reshape({3, 4}).transpose(0, 1); // not contiguous
    auto t_hw = host_to_device<int32_t>(t);
    ASSERT_FALSE(t_hw->is_contiguous());
}

TEST(ContiguousTests, IsContiguousTensorContiguous) {
    auto t = torch::arange(12, torch::kInt32); // contiguous
    auto t_hw = host_to_device<int32_t>(t);
    ASSERT_TRUE(t_hw->is_contiguous());
}