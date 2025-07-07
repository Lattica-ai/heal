#include "gtest/gtest.h"
#include "lattica_hw_api.h"

using namespace lattica_hw_api;

/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                        SET_CONST_VAL TESTS                                     ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

TEST(SetConstValTests, Basic1DInt32) {
    // start with some arbitrary data
    auto t_cpu = torch::tensor({1, 2, -3, 42}, torch::kInt32);
    auto hw_t = host_to_device<int32_t>(t_cpu);

    // set every element to 7
    set_const_val<int32_t>(hw_t, 7);
    auto out = device_to_host<int32_t>(hw_t);
    auto expected = torch::full({4}, 7, torch::kInt32);

    ASSERT_TRUE(torch::equal(out, expected));
}

TEST(SetConstValTests, ScalarInt) {
    // zero‐dimensional tensor
    auto t_cpu = torch::tensor(123, torch::kInt64);
    auto hw_t = host_to_device<int64_t>(t_cpu);

    set_const_val<int64_t>(hw_t, -999LL);
    auto out = device_to_host<int64_t>(hw_t);

    ASSERT_EQ(out.item<int64_t>(), -999LL);
}

TEST(SetConstValTests, ThreeDim) {
    // 3D tensor, shape [2, 3, 4]
    auto t_cpu = torch::arange(24, torch::kInt64).reshape({2, 3, 4});
    auto hw_t = host_to_device<int64_t>(t_cpu);

    set_const_val<int64_t>(hw_t, 555LL);
    auto out = device_to_host<int64_t>(hw_t);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                ASSERT_EQ(out[i][j][k].item<int64_t>(), 555LL);
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

TEST(SetConstValTests, Throws_OnNullTensor) {
    std::shared_ptr<DeviceTensor<int64_t>> null_ptr;
    EXPECT_THROW(
      set_const_val<int64_t>(null_ptr, 5),
      std::invalid_argument
    );
}

