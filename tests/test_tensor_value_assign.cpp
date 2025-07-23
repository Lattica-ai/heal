#include "memory_management.h"
#include "tensor_value_assign.h"
#include "gtest/gtest.h"

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


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                         PAD_SINGLE_AXIS  TESTS                                ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

// ──────────────────────────────────────────────────────────────────────────────
// Basic functionality
// ──────────────────────────────────────────────────────────────────────────────

TEST(PadSingleAxisTests, PadEnd1D) {
    torch::Tensor a = torch::tensor({1, 2, 3, 4}, torch::kInt64);
    int64_t pad = 2;
    int64_t axis = 0;

    torch::Tensor expected = torch::cat({a, torch::zeros({pad}, torch::kInt64)}, /*dim=*/0);

    auto a_hw = host_to_device<int64_t>(a);
    auto result_hw = empty<int64_t>({a.size(0) + pad});

    pad_single_axis<int64_t>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<int64_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "1-D pad-end failed.";
}


TEST(PadSingleAxisTests, PadCols2D_NegAxis) {
    // Shape [2×3]  →  pad 4 columns using axis = -1  →  [2×7]
    torch::Tensor a = torch::tensor({{1,2,3},
                                     {4,5,6}}, torch::kInt64);
    int64_t pad = 4;
    int64_t axis = -1;            // last axis

    torch::Tensor expected = torch::cat({a, torch::zeros({a.size(0), pad}, torch::kInt64)}, /*dim=*/1);

    auto a_hw = host_to_device<int64_t>(a);
    auto result_hw = empty<int64_t>({a.size(0), a.size(1) + pad});

    pad_single_axis<int64_t>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<int64_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "2-D col pad (axis=-1) failed.";
}

TEST(PadSingleAxisTests, PadCols2D_NegAxis_int32) {
    // Shape [2×3]  →  pad 4 columns using axis = -1  →  [2×7]
    torch::Tensor a = torch::tensor({{1,2,3},
                                     {4,5,6}}, torch::kInt32);
    int64_t pad = 4;
    int64_t axis = -1;            // last axis

    torch::Tensor expected = torch::cat({a, torch::zeros({a.size(0), pad}, torch::kInt32)}, /*dim=*/1);

    auto a_hw = host_to_device<int32_t>(a);
    auto result_hw = empty<int32_t>({a.size(0), a.size(1) + pad});

    pad_single_axis<int32_t>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "2-D col pad (axis=-1) failed.";
}

TEST(PadSingleAxisTests, ZeroPadNoChange) {
    torch::Tensor a = torch::arange(12, torch::kInt64).reshape({3,4});
    int64_t pad = 0;
    int64_t axis = 1;

    auto a_hw = host_to_device<int64_t>(a);
    auto result_hw = empty<int64_t>({3,4});

    pad_single_axis<int64_t>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<int64_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, a)) << "Zero-pad altered data.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Error conditions
// ──────────────────────────────────────────────────────────────────────────────

TEST(PadSingleAxisTests, NegativePadThrows) {
    torch::Tensor a = torch::tensor({1,2,3}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);
    auto result_hw  = empty<int64_t>({3});   // dummy

    EXPECT_THROW(pad_single_axis<int64_t>(a_hw, -1, 0, result_hw),
                 std::invalid_argument);
}

TEST(PadSingleAxisTests, AxisOutOfRangeThrows) {
    torch::Tensor a = torch::ones({2,2}, torch::kInt64);
    auto a_hw  = host_to_device<int64_t>(a);
    auto result_hw = empty<int64_t>({2,2});  // shape doesn’t matter

    EXPECT_THROW(pad_single_axis<int64_t>(a_hw, 1,  2, result_hw), std::invalid_argument); // > rank-1
    EXPECT_THROW(pad_single_axis<int64_t>(a_hw, 1, -3, result_hw), std::invalid_argument); // < -rank
}

TEST(PadSingleAxisTests, RankMismatchThrows) {
    torch::Tensor a = torch::ones({2,2}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);
    auto result_hw = empty<int64_t>({4});  // rank-1 instead of 2

    EXPECT_THROW(pad_single_axis<int64_t>(a_hw, 2, 0, result_hw), std::invalid_argument);
}

TEST(PadSingleAxisTests, OutputDimMismatchThrows) {
    // pad = 1 on axis 1 → result should have 4 cols but we give only 3
    torch::Tensor a = torch::tensor({{1,2,3}, {4,5,6}}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);
    auto result_hw = empty<int64_t>({2,3});  // wrong shape

    EXPECT_THROW(pad_single_axis<int64_t>(a_hw, 1, 1, result_hw), std::invalid_argument);
}
