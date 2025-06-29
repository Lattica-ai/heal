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


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                             MOVEAXIS  TESTS                                   ****
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

    moveaxis<int64_t>(a_hw, /*src=*/2, /*dst=*/0);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Move axis 2→0 failed.";
}

TEST(MoveAxisTests, MoveFirstToLast3D_Int32) {
    // shape [2,3,4]  →  move axis 0 → 2  ⇒  [3,4,2]
    torch::Tensor a = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/0, /*dst=*/2);

    auto a_hw = host_to_device<int64_t>(a);

    moveaxis<int64_t>(a_hw, /*src=*/0, /*dst=*/2);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Move axis 0→2 failed.";
}

TEST(MoveAxisTests, MoveMiddleToLast4D_Int32) {
    // shape [2,3,4,5]  →  move axis 1 → -1  ⇒  [2,4,5,3]
    torch::Tensor a = torch::arange(2*3*4*5, torch::kInt32).reshape({2,3,4,5});
    torch::Tensor expected = torch::movedim(a, /*src=*/1, /*dst=*/-1);

    auto a_hw = host_to_device<int32_t>(a);

    moveaxis<int32_t>(a_hw, /*src=*/1, /*dst=*/-1);

    torch::Tensor result = device_to_host<int32_t>(a_hw);
    ASSERT_TRUE(torch::allclose(result, expected))
        << "Move middle axis to last failed.";
}

TEST(MoveAxisTests, MoveFirstToLast3D_NegativeSrc) {
    // axis_src = -3  == 0
    torch::Tensor a = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/0, /*dst=*/2);

    auto a_hw = host_to_device<int64_t>(a);

    moveaxis<int64_t>(a_hw, /*src=*/-3, /*dst=*/-1);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Negative src axis (-rank) move failed.";
}

TEST(MoveAxisTests, NoOpWhenAxesEqual) {
    torch::Tensor a = torch::randint(0, 10, {3,4,5}, torch::kInt64);
    auto a_hw = host_to_device<int64_t>(a);

    moveaxis<int64_t>(a_hw, /*src=*/1, /*dst=*/1);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, a)) << "No-op moveaxis altered tensor.";
}

TEST(MoveAxisTests, MoveAdjacentForward) {
    // [2,3,4]  ->  move axis 1 -> 2  ==>  [2,4,3]
    torch::Tensor a        = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/1, /*dst=*/2);

    auto a_hw = host_to_device<int64_t>(a);
    moveaxis<int64_t>(a_hw, /*src=*/1, /*dst=*/2);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Adjacent forward move (1→2) failed.";
}

TEST(MoveAxisTests, MoveAdjacentBackward) {
    // [2,3,4]  ->  move axis 2 -> 1  ==>  [2,4,3]
    torch::Tensor a        = torch::arange(2*3*4, torch::kInt64).reshape({2,3,4});
    torch::Tensor expected = torch::movedim(a, /*src=*/2, /*dst=*/1);

    auto a_hw = host_to_device<int64_t>(a);
    moveaxis<int64_t>(a_hw, /*src=*/2, /*dst=*/1);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::equal(result, expected))
        << "Adjacent backward move (2→1) failed.";
}

TEST(MoveAxisTests, NonContiguousStrides) {
    // Start with contiguous [2,3,4], transpose to make non-contiguous, then move axis 0 -> 2
    torch::Tensor a = torch::arange(24, torch::kInt64).reshape({2,3,4}).transpose(0,1); // shape [3,2,4]
    torch::Tensor expected = torch::movedim(a, /*src=*/0, /*dst=*/2);                     // shape [2,4,3]

    auto a_hw = host_to_device<int64_t>(a);
    moveaxis<int64_t>(a_hw, /*src=*/0, /*dst=*/2);

    torch::Tensor result = device_to_host<int64_t>(a_hw);
    ASSERT_TRUE(torch::allclose(result, expected))
        << "Moveaxis with non-contiguous strides failed.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Error conditions
// ──────────────────────────────────────────────────────────────────────────────

TEST(MoveAxisTests, NullPointerThrows) {
    EXPECT_THROW(
        moveaxis<int32_t>(/*a=*/nullptr, 0, 1),
        std::invalid_argument
    );
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
