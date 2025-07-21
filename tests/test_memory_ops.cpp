#include "memory_virtual_ops.h"
#include "device_memory.h"
#include "gtest/gtest.h"

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


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                             MOVEAXIS  TESTS                                    ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

// ──────────────────────────────────────────────────────────────────────────────
// Basic functionality
// ──────────────────────────────────────────────────────────────────────────────

 TEST(MoveAxisTests, MoveLastToFirst3D_Int32) {
    // shape [2,3,4]  →  move axis 2 → 0  ⇒  [4,2,3]
    torch::Tensor a = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/2, /*dst=*/0);

    auto a_hw = host_to_device<int64_t>(a);

    a_hw = moveaxis<int64_t>(a_hw, /*src=*/2, /*dst=*/0);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Move axis 2→0 failed.";
}

TEST(MoveAxisTests, MoveFirstToLast3D_Int32) {
    // shape [2,3,4]  →  move axis 0 → 2  ⇒  [3,4,2]
    torch::Tensor a = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/0, /*dst=*/2);

    auto a_hw = host_to_device<int64_t>(a);

    a_hw = moveaxis<int64_t>(a_hw, /*src=*/0, /*dst=*/2);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Move axis 0→2 failed.";
}

TEST(MoveAxisTests, MoveMiddleToLast4D_Int32) {
    // shape [2,3,4,5]  →  move axis 1 → -1  ⇒  [2,4,5,3]
    torch::Tensor a = torch::arange(2*3*4*5, torch::kInt32).reshape({2,3,4,5});
    torch::Tensor expected = torch::movedim(a, /*src=*/1, /*dst=*/-1);

    auto a_hw = host_to_device<int32_t>(a);

    a_hw = moveaxis<int32_t>(a_hw, /*src=*/1, /*dst=*/-1);

    torch::Tensor result = device_to_host<int32_t>(a_hw);
    ASSERT_TRUE(torch::allclose(result, expected))
        << "Move middle axis to last failed.";
}

TEST(MoveAxisTests, MoveFirstToLast3D_NegativeSrc) {
    // axis_src = -3  == 0
    torch::Tensor a = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/0, /*dst=*/2);

    auto a_hw = host_to_device<int64_t>(a);

    a_hw = moveaxis<int64_t>(a_hw, /*src=*/-3, /*dst=*/-1);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Negative src axis (-rank) move failed.";
}

TEST(MoveAxisTests, NoOpWhenAxesEqual) {
    torch::Tensor a = torch::randint(0, 10, {3,4,5}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);

    a_hw = moveaxis<int64_t>(a_hw, /*src=*/1, /*dst=*/1);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, a)) << "No-op moveaxis altered tensor.";
}

TEST(MoveAxisTests, MoveAdjacentForward) {
    // [2,3,4]  ->  move axis 1 -> 2  ==>  [2,4,3]
    torch::Tensor a        = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/1, /*dst=*/2);

    auto a_hw = host_to_device<int64_t>(a);
    a_hw = moveaxis<int64_t>(a_hw, /*src=*/1, /*dst=*/2);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Adjacent forward move (1→2) failed.";
}

TEST(MoveAxisTests, MoveAdjacentBackward) {
    // [2,3,4]  ->  move axis 2 -> 1  ==>  [2,4,3]
    torch::Tensor a        = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/2, /*dst=*/1);

    auto a_hw = host_to_device<int64_t>(a);
    a_hw = moveaxis<int64_t>(a_hw, /*src=*/2, /*dst=*/1);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Adjacent backward move (2→1) failed.";
}

TEST(MoveAxisTests, NonContiguousStrides) {
    // Start with contiguous [2,3,4], transpose to make non-contiguous, then move axis 0 -> 2
    torch::Tensor a = torch::arange(24, torch::kInt64).reshape({2,3,4}).transpose(0,1); // shape [3,2,4]
    torch::Tensor expected = torch::movedim(a, /*src=*/0, /*dst=*/2);                     // shape [2,4,3]

    auto a_hw = host_to_device<int64_t>(a);
    a_hw = moveaxis<int64_t>(a_hw, /*src=*/0, /*dst=*/2);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::allclose(result, expected))
        << "Moveaxis with non-contiguous strides failed.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Error conditions
// ──────────────────────────────────────────────────────────────────────────────

TEST(MoveAxisTests, NullPointerThrows) {
    EXPECT_THROW(moveaxis<int32_t>(/*a=*/nullptr, 0, 1), std::invalid_argument);
}

TEST(MoveAxisTests, SrcAxisOutOfRangeThrows) {
    torch::Tensor a = torch::randint(0, 10, {2,2,2}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);

    EXPECT_THROW(moveaxis<int64_t>(a_hw,  3, 0), std::invalid_argument);  // > rank-1
    EXPECT_THROW(moveaxis<int64_t>(a_hw, -4, 0), std::invalid_argument);  // < -rank
}

TEST(MoveAxisTests, DstAxisOutOfRangeThrows) {
    torch::Tensor a = torch::randint(0, 10, {2,2,2}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);

    EXPECT_THROW(moveaxis<int64_t>(a_hw, 1,  3), std::invalid_argument);
    EXPECT_THROW(moveaxis<int64_t>(a_hw, 1, -4), std::invalid_argument);
}


/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                             GET_SLICE TESTS                                    ****
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

/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                             FLATTEN TESTS                                      ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

// ──────────────────────────────────────────────────────────────────────────────
// Basic test: flatten a 2D tensor into 1D
// ──────────────────────────────────────────────────────────────────────────────

TEST(FlattenTests, FlattenAllDims2D) {
    // a = [[ 0,  1,  2],
    //      [ 3,  4,  5]]
    auto a = torch::arange(0, 6, torch::kInt64).reshape({2,3});
    // flatten dims [0..1] → single dim of size 6
    auto a_hw = host_to_device<int64_t>(a);
    auto out_hw = flatten<int64_t>(a_hw, 0, 1);
    auto out = device_to_host<int64_t>(out_hw);

    // expected = a.view({6})
    auto expected = a.view({6});
    ASSERT_TRUE(torch::equal(out, expected))
        << "Flattening both dims into one did not match expected shape or values.";
    ASSERT_EQ(out.dim(), 1)
        << "Result should be a 1D tensor.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Basic test: flatten a 2D tensor into 1D int32_t
// ────────────────────────────────────────────────────────────────────────────────

TEST(FlattenTests, FlattenAllDims2DInt32) {
    auto a = torch::arange(0, 6, torch::kInt32).reshape({2,3});
    auto a_hw = host_to_device<int32_t>(a);

    auto out_hw = flatten<int32_t>(a_hw, 0, 1);
    auto out = device_to_host<int32_t>(out_hw);

    auto expected = a.reshape({6});
    ASSERT_TRUE(torch::equal(out, expected))
        << "Flatten should preserve element order.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Flatten only a single dimension (no-op)
// ──────────────────────────────────────────────────────────────────────────────

TEST(FlattenTests, FlattenSingleDimNoOp) {
    // a = [[1,2],[3,4]]
    auto a = torch::tensor({{1,2},{3,4}}, torch::kInt64);
    // flatten only dim 1..1 → shape unchanged
    auto a_hw = host_to_device<int64_t>(a);
    auto out_hw = flatten<int64_t>(a_hw, 1, 1);
    auto out = device_to_host<int64_t>(out_hw);

    ASSERT_EQ(out.sizes(), a.sizes())
        << "Flattening a single dimension range should leave the shape unchanged.";
    ASSERT_TRUE(torch::equal(out, a))
        << "Values must remain identical when flattening a single-dim range.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Flatten leading dims in a 3D tensor
// ──────────────────────────────────────────────────────────────────────────────

TEST(FlattenTests, FlattenLeadingDims3D) {
    // a.shape = [2,3,4]
    auto a = torch::arange(0, 24, torch::kInt64).reshape({2,3,4});
    // flatten dims [0..1] → new shape [6,4]
    auto a_hw = host_to_device<int64_t>(a);
    auto out_hw = flatten<int64_t>(a_hw, 0, 1);
    auto out = device_to_host<int64_t>(out_hw);

    auto expected = a.reshape({6,4});
    ASSERT_EQ(out.sizes(), expected.sizes())
        << "Flattening first two dims did not yield [6,4].";
    ASSERT_TRUE(torch::equal(out, expected))
        << "Flattened values do not match expected ordering.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Flatten trailing dims in a 4D tensor
// ──────────────────────────────────────────────────────────────────────────────

TEST(FlattenTests, FlattenTrailingDims4D) {
    // a.shape = [2, 3, 4, 5]
    auto a = torch::arange(0, 2*3*4*5, torch::kInt64).reshape({2,3,4,5});
    // flatten dims [2..3] → new shape [2,3,20]
    auto a_hw = host_to_device<int64_t>(a);
    auto out_hw = flatten<int64_t>(a_hw, 2, 3);
    auto out = device_to_host<int64_t>(out_hw);

    auto expected = a.reshape({2,3,20});
    ASSERT_EQ(out.sizes(), expected.sizes())
        << "Flattening last two dims did not yield [2,3,20].";
    ASSERT_TRUE(torch::equal(out, expected))
        << "Flattened values do not match expected ordering.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Flatten with negative indices
// ──────────────────────────────────────────────────────────────────────────────

TEST(FlattenTests, FlattenNegativeIndices) {
    // a.shape = [4,5,6]
    auto a = torch::arange(0, 4*5*6, torch::kInt64).reshape({4,5,6});
    // flatten dims [-3..-2] == [0..1] → [20,6]
    auto a_hw = host_to_device<int64_t>(a);
    auto out_hw = flatten<int64_t>(a_hw, -3, -2);
    auto out = device_to_host<int64_t>(out_hw);

    auto expected = a.reshape({20,6});
    ASSERT_EQ(out.sizes(), expected.sizes())
        << "Flattening with negative indices did not yield expected shape.";
    ASSERT_TRUE(torch::equal(out, expected))
        << "Flattened values via negative indices do not match expected.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Flatten including singleton dims
// ──────────────────────────────────────────────────────────────────────────────

TEST(FlattenTests, FlattenIncludingOnes) {
    // shape = [2,1,3,1,4]
    auto a = torch::arange(0, 2*1*3*1*4, torch::kInt64).reshape({2,1,3,1,4});
    // flatten dims [1..3] (1 * 3 * 1 = 3) → new shape [2,3,4]
    auto a_hw = host_to_device<int64_t>(a);
    auto out_hw = flatten<int64_t>(a_hw, 1, 3);
    auto out = device_to_host<int64_t>(out_hw);

    auto expected = a.reshape({2,3,4});
    ASSERT_EQ(out.sizes(), (std::vector<int64_t>{2,3,4}))
        << "Flattening over singleton dims should collapse [1,3,1] → 3.";
    ASSERT_TRUE(torch::equal(out, expected))
        << "Values after flattening singleton dims aren’t in the expected order.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Error cases
// ──────────────────────────────────────────────────────────────────────────────

TEST(FlattenTests, ThrowsOnNonContiguousInput) {
    // Create a tensor and make it non-contiguous (transpose)
    auto a = torch::arange(0, 12, torch::kInt64).reshape({3, 4}).transpose(0, 1);
    ASSERT_FALSE(a.is_contiguous());  // PyTorch: sanity check

    auto a_hw = host_to_device<int64_t>(a);

    // flatten should throw
    EXPECT_THROW({
        flatten<int64_t>(a_hw, 0, 1);
    }, std::runtime_error);
}

TEST(FlattenTests, InvalidStartEndThrows) {
    auto a = torch::randint(0, 10, {3,4,5}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);

    // start_dim < 0 after wrap
    EXPECT_THROW(flatten<int64_t>(a_hw, -4, 1), std::invalid_argument);

    // end_dim < start_dim
    EXPECT_THROW(flatten<int64_t>(a_hw, 2, 1), std::invalid_argument);

    // end_dim >= ndim
    EXPECT_THROW(flatten<int64_t>(a_hw, 1, 3), std::invalid_argument);
}

TEST(FlattenTests, OutOfRangeDimsThrows) {
    auto a = torch::zeros({5,5}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);

    // start_dim too large
    EXPECT_THROW(flatten<int64_t>(a_hw, 2, 2), std::invalid_argument);

    // end_dim too small
    EXPECT_THROW(flatten<int64_t>(a_hw, 0, -3), std::invalid_argument);
}


/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                             RESHAPE TESTS                                      ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

TEST(ReshapeTests, ReshapeFunctionality) {

    // Create initial tensor [2, 2, 3]
    torch::Tensor c_cpu = torch::tensor(
        {{{1, 2, 3}, {4, 5, 6}},
         {{7, 8, 9}, {10, 11, 12}}},
        torch::dtype(torch::kInt32)
    );

    auto c_hw = lattica_hw_api::host_to_device<int32_t>(c_cpu);

    // Reshape to [6, 2]
    c_hw = reshape(c_hw, {6, 2});
    // Validate content integrity after reshaping
    torch::Tensor expected_after_reshape1 = torch::tensor(
        {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}},
        torch::dtype(torch::kInt32)
    ); // [6, 2]
    torch::Tensor result_after_reshape1 = lattica_hw_api::device_to_host<int32_t>(c_hw);
    ASSERT_TRUE(torch::equal(result_after_reshape1, expected_after_reshape1)) << "Content mismatch after reshape to [6, 2].";


    // Reshape to [3, 4]
    c_hw = reshape(c_hw, {3, 4});
    // Validate content integrity after reshaping
    torch::Tensor expected_after_reshape2 = torch::tensor(
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        torch::dtype(torch::kInt32)
    ); // [3, 4]
    torch::Tensor result_after_reshape2 = lattica_hw_api::device_to_host<int32_t>(c_hw);
    ASSERT_TRUE(torch::equal(result_after_reshape2, expected_after_reshape2)) << "Content mismatch after reshape to [3, 4].";
}