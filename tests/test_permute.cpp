#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

torch::Tensor permute_expected(
    const torch::Tensor& a,
    const torch::Tensor& perms,
    int64_t elementwise_axis,
    int64_t perm_axis
) {
    // Move the elementwise and permutation axes to the end (last two axes)
    auto a_moved = torch::movedim(a.clone(), {elementwise_axis, perm_axis}, {-2, -1});  // shape: [..., l, m]

    // Build new shape with leading 1s to match a_moved
    std::vector<int64_t> new_shape(a_moved.dim(), 1);
    new_shape[new_shape.size() - 2] = perms.size(0);  // l
    new_shape[new_shape.size() - 1] = perms.size(1);  // m
    auto perms_exp = perms.view(new_shape).expand_as(a_moved).to(torch::kLong);  // [..., l, m]

    // Apply permutation using gather
    auto result_moved = torch::take_along_dim(a_moved, perms_exp, -1);  // gather along last dim (m)

    // Move the axes back to original positions
    return torch::movedim(result_moved, {-2, -1}, {elementwise_axis, perm_axis});
}

void run_permute_case(
    const torch::Tensor& a,
    const torch::Tensor& perms,
    int64_t elementwise_axis,
    int64_t perm_axis
) {
    auto a_hw = host_to_device<int32_t>(a);
    auto perms_hw = host_to_device<int32_t>(perms);
    auto result_hw = empty<int32_t>(a.sizes().vec());

    permute<int32_t>(a_hw, perms_hw, result_hw, elementwise_axis, perm_axis);
    auto result = device_to_host<int32_t>(result_hw);

    auto expected = permute_expected(a, perms, elementwise_axis, perm_axis);
    ASSERT_TRUE(torch::equal(result, expected)) << "Permutation result mismatch.";
}

TEST(PermuteTests, Rank4_PermuteDims0And1) {
    auto a = torch::randint(0, 100, {2, 3, 4, 5}, torch::kInt32);  // [l, m, r, k]
    auto perms = torch::tensor({{2, 1, 0}, {0, 2, 1}}, torch::kInt32);  // [l=2, m=3]
    run_permute_case(a, perms, 0, 1);
}

TEST(PermuteTests, Rank5_PermuteDims0And3) {
    auto a = torch::randint(0, 100, {2, 2, 2, 3, 5}, torch::kInt32); // [A, A, A, l, m]
    auto perms = torch::tensor({{1, 0, 2}, {2, 1, 0}}, torch::kInt32);  // [l=2, m=3]
    run_permute_case(a, perms, 0, 3);
}

TEST(PermuteTests, PermuteInnerDims) {
    auto a = torch::randint(0, 50, {3, 4, 5}, torch::kInt32); // [l=3, m=4, k=5]
    auto perms = torch::tensor({{3, 2, 1, 0}, {0, 2, 3, 1}, {1, 0, 3, 2}}, torch::kInt32); // [3, 4]
    run_permute_case(a, perms, 0, 1);
}

TEST(PermuteTests, InvalidDimsThrows) {
    auto a = torch::randint(0, 10, {2, 3, 4}, torch::kInt32);
    auto perms = torch::randint(0, 3, {2, 3}, torch::kInt32);
    auto a_hw = host_to_device<int32_t>(a);
    auto perms_hw = host_to_device<int32_t>(perms);
    auto result_hw = empty<int32_t>({2, 3, 4});

    EXPECT_THROW(permute<int32_t>(a_hw, perms_hw, result_hw, 3, 1), std::invalid_argument);
    EXPECT_THROW(permute<int32_t>(a_hw, perms_hw, result_hw, 1, 1), std::invalid_argument);
    EXPECT_THROW(permute<int32_t>(a_hw, perms_hw, result_hw, 0, 3), std::invalid_argument);
}

TEST(PermuteTests, PermutationOutOfBoundsThrows) {
    auto a = torch::randint(0, 10, {2, 3, 4}, torch::kInt32);
    auto perms = torch::tensor({{0, 1, 3}, {1, 2, 0}}, torch::kInt32);  // 3 is OOB
    auto a_hw = host_to_device<int32_t>(a);
    auto perms_hw = host_to_device<int32_t>(perms);
    auto result_hw = empty<int32_t>({2, 3, 4});

    EXPECT_THROW(permute<int32_t>(a_hw, perms_hw, result_hw, 0, 1), std::out_of_range);
}

TEST(PermuteTests, PermuteDim1_ElementwiseDim0) {
    // shape: [5, 3, 7]
    auto a = torch::randint(0, 100, {5, 3, 7}, torch::kInt32);
    auto perms = torch::tensor({
        {2, 1, 0},
        {0, 2, 1},
        {1, 0, 2},
        {2, 0, 1},
        {0, 1, 2}
    }, torch::kInt32);  // shape: [5, 3]
    run_permute_case(a, perms, /*elementwise_axis=*/0, /*perm_axis=*/1);
}

TEST(PermuteTests, PermuteDim1_ElementwiseDim2) {
    // shape: [3, 4, 2]
    auto a = torch::randint(0, 100, {3, 4, 2}, torch::kInt32);
    auto perms = torch::tensor({
        {1, 2, 0, 3},
        {3, 0, 1, 2}
    }, torch::kInt32);  // shape: [2, 4]
    run_permute_case(a, perms, /*elementwise_axis=*/2, /*perm_axis=*/1);
}
