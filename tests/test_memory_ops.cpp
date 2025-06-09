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


/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                             GET_SLICE TESTS                                     ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

// ──────────────────────────────────────────────────────────────────────────────
// Basic 2D slice (contiguous, unit step)
// ──────────────────────────────────────────────────────────────────────────────

TEST(GetSliceTests, Basic2DSlice) {
    // a = [[ 0,  1,  2,  3],
    //      [ 4,  5,  6,  7]]
    auto a = torch::arange(0, 8, torch::kInt64).reshape({2,4});
    std::vector<SliceArg> slices = {
        Slice(0, 2),      // take rows [0,1]
        Slice(1, 4)       // take cols [1,2,3]
    };
    auto a_hw   = host_to_device<int64_t>(a);
    auto out_hw = get_slice<int64_t>(a_hw, slices);
    auto out    = device_to_host<int64_t>(out_hw);

    // expected = a.slice(0,0,2).slice(1,1,4)
    auto expected = a.slice(0, 0, 2).slice(1, 1, 4);
    ASSERT_TRUE(torch::equal(out, expected))
        << "Basic 2D slice did not match expected values.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Slice with step > 1
// ──────────────────────────────────────────────────────────────────────────────

TEST(GetSliceTests, StepSlice2D) {
    // a = [0,1,2,3,4,5]
    auto a = torch::arange(0, 6, torch::kInt64);
    std::vector<SliceArg> slices = {
        Slice(1, 6, 2)   // take indices [1,3,5]
    };
    auto a_hw   = host_to_device<int64_t>(a);
    auto out_hw = get_slice<int64_t>(a_hw, slices);
    auto out    = device_to_host<int64_t>(out_hw);

    // expected = a.slice(0,1,6,2)
    auto expected = a.slice(0, 1, 6, 2);
    ASSERT_TRUE(torch::equal(out, expected))
        << "1D slice with step did not match expected values.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Full collapse to scalar
// ──────────────────────────────────────────────────────────────────────────────

TEST(GetSliceTests, FullCollapseScalar) {
    // a is 3D tensor [[ [7,8], [9,10] ]]
    auto a = torch::tensor({{{7,8},{9,10}}}, torch::kInt64);
    // collapse all dims: pick element [0][1][1] → 10
    std::vector<SliceArg> slices = {
        int64_t(0),       // depth
        int64_t(1),       // row
        int64_t(1)        // col
    };
    auto a_hw   = host_to_device<int64_t>(a);
    auto out_hw = get_slice<int64_t>(a_hw, slices);
    auto out    = device_to_host<int64_t>(out_hw);

    // out should be a scalar tensor == 10
    ASSERT_EQ(out.item<int64_t>(), 10)
        << "Full collapse to scalar returned wrong value.";
    ASSERT_EQ(out.sizes().size(), 0)
        << "Scalar collapse should produce a 0-d tensor.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Mixed index and slice in 3D tensor
// ──────────────────────────────────────────────────────────────────────────────

TEST(GetSliceTests, MixedIndexAndSlice3D) {
    // a dimensions [2,3,4]
    auto a = torch::arange(0, 24, torch::kInt64).reshape({2,3,4});
    // Collapse dim 0 at index 1 → shape [3,4], then slice dim1 [0,3) step=2, dim2 [1,4)
    std::vector<SliceArg> slices = {
        int64_t(1),       // pick second block [3x4]
        Slice(0, 3, 2),   // take rows [0,2]
        Slice(1, 4)       // take cols [1,2,3]
    };
    auto a_hw   = host_to_device<int64_t>(a);
    auto out_hw = get_slice<int64_t>(a_hw, slices);
    auto out    = device_to_host<int64_t>(out_hw);

    // expected = a[1].slice(0,0,3,2).slice(1,1,4)
    auto expected = a[1].slice(0, 0, 3, 2).slice(1, 1, 4);
    ASSERT_TRUE(torch::equal(out, expected))
        << "Mixed index + slice in 3D did not match expected values.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Non-contiguous input (transpose) + slicing
// ──────────────────────────────────────────────────────────────────────────────

TEST(GetSliceTests, NonContiguousInputTransposeSlice) {
    // base = [[0,1,2],
    //         [3,4,5],
    //         [6,7,8]]
    auto base = torch::arange(0, 9, torch::kInt64).reshape({3,3});
    auto a    = base.t();  // transpose → shape [3,3], non-contiguous

    // slice rows [1,3) → rows 1,2
    //       cols [0,3) step=2 → cols 0,2
    std::vector<SliceArg> slices = {
        Slice(1, 3),
        Slice(0, 3, 2)
    };

    auto a_hw   = host_to_device<int64_t>(a);
    auto out_hw = get_slice<int64_t>(a_hw, slices);
    auto out    = device_to_host<int64_t>(out_hw);

    auto expected = a.slice(0, 1, 3).slice(1, 0, 3, 2);
    ASSERT_TRUE(torch::equal(out, expected))
        << "Slicing a non-contiguous (transposed) tensor failed.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Step larger than span → single element
// ──────────────────────────────────────────────────────────────────────────────

TEST(GetSliceTests, StepGreaterThanSpanYieldsSingleElement) {
    // a = [0,1,2,3,4,5,6,7,8,9]
    auto a = torch::arange(0, 10, torch::kInt64);
    // slice from 3→5 with step=10 → only index 3
    std::vector<SliceArg> slices = {
        Slice(3, 5, 10)
    };

    auto a_hw   = host_to_device<int64_t>(a);
    auto out_hw = get_slice<int64_t>(a_hw, slices);
    auto out    = device_to_host<int64_t>(out_hw);

    // should produce a length-1 tensor [3]
    ASSERT_EQ(out.size(0), 1)
        << "Expected exactly one element when step > span.";
    ASSERT_EQ(out.item<int64_t>(), 3)
        << "Expected the single element to be the start index (3).";
}

// ──────────────────────────────────────────────────────────────────────────────
// Int32 basic slice
// ──────────────────────────────────────────────────────────────────────────────

TEST(GetSliceTests, Int32Basic2DSlice) {
    // a = [[ 10,  20,  30,  40],
    //      [ 50,  60,  70,  80]]
    auto a = torch::tensor({{10,20,30,40},{50,60,70,80}}, torch::kInt32);
    std::vector<SliceArg> slices = {
        Slice(0, 2),      // take both rows
        Slice(1, 3)       // take cols [1,2]
    };
    auto a_hw   = host_to_device<int32_t>(a);
    auto out_hw = get_slice<int32_t>(a_hw, slices);
    auto out    = device_to_host<int32_t>(out_hw);

    // expected = a.slice(0,0,2).slice(1,1,3)
    auto expected = a.slice(0, 0, 2).slice(1, 1, 3);
    ASSERT_TRUE(torch::equal(out, expected))
        << "Int32 basic 2D slice did not match expected values.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Error conditions
// ──────────────────────────────────────────────────────────────────────────────

TEST(GetSliceTests, MismatchedRankThrows) {
    auto a = torch::randint(0, 10, {2,2}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);
    // only one slice for a 2D tensor → error
    std::vector<SliceArg> slices = { Slice(0,1) };
    EXPECT_THROW(get_slice<int64_t>(a_hw, slices), std::invalid_argument);
}

TEST(GetSliceTests, IndexOutOfRangeThrows) {
    auto a = torch::zeros({3,3}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);
    // index 5 ≥ dim size 3
    std::vector<SliceArg> slices = { int64_t(5), Slice(0,1) };
    EXPECT_THROW(get_slice<int64_t>(a_hw, slices), std::out_of_range);
}

TEST(GetSliceTests, InvalidSliceStartEndThrows) {
    auto a = torch::zeros({4}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);
    // end ≤ start
    std::vector<SliceArg> slices1 = { Slice(3, 2) };
    EXPECT_THROW(get_slice<int64_t>(a_hw, slices1), std::invalid_argument);
    // end > dim size
    std::vector<SliceArg> slices2 = { Slice(0, 5) };
    EXPECT_THROW(get_slice<int64_t>(a_hw, slices2), std::invalid_argument);
}

TEST(GetSliceTests, InvalidStepThrows) {
    auto a = torch::zeros({5}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);
    // step = 0
    std::vector<SliceArg> slices = { Slice(0, 5, 0) };
    EXPECT_THROW(get_slice<int64_t>(a_hw, slices), std::invalid_argument);
}
