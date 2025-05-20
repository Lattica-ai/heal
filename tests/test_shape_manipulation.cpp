#include "gtest/gtest.h"
#include "lattica_hw_api.h"

using namespace lattica_hw_api;

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

TEST(PadSingleAxisTests, PadEnd1D_Int32) {
    torch::Tensor a = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    int64_t pad  = 2;              // append two zeros
    int64_t axis = 0;

    torch::Tensor expected =
        torch::cat({a, torch::zeros({pad}, torch::kInt32)}, /*dim=*/0);

    auto a_hw      = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>({a.size(0) + pad});

    pad_single_axis<int32_t>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "1-D pad-end failed.";
}

TEST(PadSingleAxisTests, PadRows2D_Double32) {
    // Shape [2×3]  →  pad one extra row  →  [3×3]
    torch::Tensor a = torch::tensor({{1,2,3},
                                     {4,5,6}}, torch::kFloat64);
    int64_t pad  = 1;
    int64_t axis = 0;

    torch::Tensor expected =
        torch::cat({a, torch::zeros({pad, a.size(1)}, torch::kFloat64)}, axis);

    auto a_hw      = host_to_device<double>(a);
    auto result_hw = allocate_on_hardware<double>({a.size(0) + pad, a.size(1)});

    pad_single_axis<double>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<double>(result_hw);
    ASSERT_TRUE(torch::allclose(result, expected))
        << "2-D row-padding failed.";
}

TEST(PadSingleAxisTests, PadCols2D_NegAxis) {
    // Shape [2×3]  →  pad 4 columns using axis = -1  →  [2×7]
    torch::Tensor a = torch::tensor({{1,2,3},
                                     {4,5,6}}, torch::kInt32);
    int64_t pad  = 4;
    int64_t axis = -1;            // last axis

    torch::Tensor expected =
        torch::cat({a, torch::zeros({a.size(0), pad}, torch::kInt32)}, /*dim=*/1);

    auto a_hw      = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>({a.size(0), a.size(1) + pad});

    pad_single_axis<int32_t>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "2-D col pad (axis=-1) failed.";
}

TEST(PadSingleAxisTests, HighRankStress) {
    // Input shape [2,3,4,5]  →  pad 3 slices on axis -2  →  [2,3,7,5]
    torch::Tensor a = torch::arange(120, torch::kInt32)
                          .reshape({2,3,4,5});
    int64_t pad  = 3;
    int64_t axis = -2;

    torch::Tensor zeros = torch::zeros(
        {a.size(0), a.size(1), pad, a.size(3)}, torch::kInt32);
    torch::Tensor expected = torch::cat({a, zeros}, axis);

    auto a_hw      = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>(
                        {a.size(0), a.size(1), a.size(2)+pad, a.size(3)});

    pad_single_axis<int32_t>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "High-rank stress pad failed.";
}

TEST(PadSingleAxisTests, ZeroPadNoChange) {
    torch::Tensor a = torch::arange(12, torch::kInt32).reshape({3,4});
    int64_t pad  = 0;
    int64_t axis = 1;

    auto a_hw      = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>({3,4});

    pad_single_axis<int32_t>(a_hw, pad, axis, result_hw);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, a)) << "Zero-pad altered data.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Error conditions
// ──────────────────────────────────────────────────────────────────────────────

TEST(PadSingleAxisTests, NegativePadThrows) {
    torch::Tensor a = torch::tensor({1,2,3}, torch::kInt32);
    auto a_hw       = host_to_device<int32_t>(a);
    auto result_hw  = allocate_on_hardware<int32_t>({3});   // dummy

    EXPECT_THROW(pad_single_axis<int32_t>(a_hw, -1, 0, result_hw),
                 std::invalid_argument);
}

TEST(PadSingleAxisTests, AxisOutOfRangeThrows) {
    torch::Tensor a = torch::ones({2,2}, torch::kInt32);
    auto a_hw      = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>({2,2});  // shape doesn’t matter

    EXPECT_THROW(pad_single_axis<int32_t>(a_hw, 1,  2, result_hw), std::invalid_argument); // > rank-1
    EXPECT_THROW(pad_single_axis<int32_t>(a_hw, 1, -3, result_hw), std::invalid_argument); // < -rank
}

TEST(PadSingleAxisTests, RankMismatchThrows) {
    torch::Tensor a = torch::ones({2,2}, torch::kInt32);
    auto a_hw      = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>({4});  // rank-1 instead of 2

    EXPECT_THROW(pad_single_axis<int32_t>(a_hw, 2, 0, result_hw), std::invalid_argument);
}

TEST(PadSingleAxisTests, OutputDimMismatchThrows) {
    // pad = 1 on axis 1 → result should have 4 cols but we give only 3
    torch::Tensor a = torch::tensor({{1,2,3}, {4,5,6}}, torch::kInt32);
    auto a_hw      = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>({2,3});  // wrong shape

    EXPECT_THROW(pad_single_axis<int32_t>(a_hw, 1, 1, result_hw), std::invalid_argument);
}

TEST(PadSingleAxisTests, NullPointersThrow) {
    // nullptr input
    auto result_hw = allocate_on_hardware<int32_t>({1});
    EXPECT_THROW(pad_single_axis<int32_t>(/*a=*/nullptr, 0, 0, result_hw),
                 std::invalid_argument);
}
