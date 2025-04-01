#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

TEST(ContiguousTests, MakesTensorContiguous) {
    auto t = torch::arange(12, torch::kInt32).reshape({3, 4}).transpose(0, 1);  // not contiguous
    auto hw = host_to_device<int32_t>(t);
    auto hw_contig = make_contiguous<int32_t>(hw);
    auto back = device_to_host<int32_t>(hw_contig);
    ASSERT_TRUE(torch::equal(back, t));
}

TEST(ContiguousTests, ReturnsSameIfAlreadyContiguous) {
    auto t = torch::randint(0, 60000, {5, 6}, torch::kInt64);
    auto hw = host_to_device<int64_t>(t);
    auto result = make_contiguous<int64_t>(hw);
    ASSERT_EQ(hw.get(), result.get());  // Same pointer
}
