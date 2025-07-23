#include "ntt.h"
#include "memory_management.h"
#include "gtest/gtest.h"

using namespace lattica_hw_api;

TEST(NTTTests, PerformNTTAndVerifyRestorationTorch) {
    // Input tensor a: [1, 4, 1, 2] → l = 1, m = 4, r = 1, k = 2
    torch::Tensor a_cpu = torch::tensor(
        {{{{1, 2}}, {{3, 4}}, {{5, 6}}, {{7, 8}}}},
        torch::dtype(torch::kInt32)
    ); // shape: [1, 4, 1, 2]

    // Parameters
    torch::Tensor p_cpu = torch::tensor({17, 257}, torch::dtype(torch::kInt32));         // [k]
    torch::Tensor m_inv_cpu = torch::tensor({13, 193}, torch::dtype(torch::kInt32));     // [k]
    torch::Tensor perm_cpu = torch::tensor({0, 2, 1, 3}, torch::dtype(torch::kInt32));   // [m]

    // twiddles and inv_twiddles now [k, m]
    torch::Tensor twiddles_cpu = torch::tensor({
        {1, 4, 2, 8},
        {1, 16, 4, 64}
    }, torch::dtype(torch::kInt32));  // [2, 4]

    torch::Tensor inv_twiddles_cpu = torch::tensor({
        {1, 13, 9, 15},
        {1, 241, 193, 253}
    }, torch::dtype(torch::kInt32));  // [2, 4]


    // Upload to hardware
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto p_hw = host_to_device<int32_t>(p_cpu);
    auto m_inv_hw = host_to_device<int32_t>(m_inv_cpu);
    auto perm_hw = host_to_device<int32_t>(perm_cpu);
    auto twiddles_hw = host_to_device<int32_t>(twiddles_cpu);
    auto inv_twiddles_hw = host_to_device<int32_t>(inv_twiddles_cpu);

    // Allocate result and restoration buffers
    auto result_hw = empty<int32_t>({1, 4, 1, 2});
    auto restored_hw = empty<int32_t>({1, 4, 1, 2});

    int64_t axis = -3;  // Axis of n

    // Perform NTT and inverse NTT
    ntt<int32_t, int32_t>(a_hw, p_hw, perm_hw, twiddles_hw, nullptr, nullptr, axis, false, result_hw);
    intt<int32_t>(result_hw, p_hw, perm_hw, inv_twiddles_hw, m_inv_hw, nullptr, nullptr, restored_hw);

    // Download result
    torch::Tensor restored_cpu = device_to_host<int32_t>(restored_hw);

    // Assert exact match
    ASSERT_TRUE(torch::equal(restored_cpu, a_cpu))
        << "Restored input does not match the original input.\n"
        << "Expected:\n" << a_cpu << "\nActual:\n" << restored_cpu;
}


TEST(NTTTests, PerformNTTOptimisedDimAndVerifyRestorationTorch) {
    // Input tensor a: [1, 4, 1, 2] → l = 1, m = 4, r = 1, k = 2
    torch::Tensor a_cpu = torch::tensor(
        {{{{1, 2}}, {{3, 4}}, {{5, 6}}, {{7, 8}}}},
        torch::dtype(torch::kInt32)
    ); // shape: [1, 4, 1, 2]

    // Parameters
    torch::Tensor p_cpu = torch::tensor({17, 257}, torch::dtype(torch::kInt32));         // [k]
    torch::Tensor m_inv_cpu = torch::tensor({13, 193}, torch::dtype(torch::kInt32));     // [k]
    torch::Tensor perm_cpu = torch::tensor({0, 2, 1, 3}, torch::dtype(torch::kInt32));   // [m]

    // twiddles and inv_twiddles now [k, m]
    torch::Tensor twiddles_cpu = torch::tensor({
        {1, 4, 2, 8},
        {1, 16, 4, 64}
    }, torch::dtype(torch::kInt32));  // [2, 4]

    torch::Tensor inv_twiddles_cpu = torch::tensor({
        {1, 13, 9, 15},
        {1, 241, 193, 253}
    }, torch::dtype(torch::kInt32));  // [2, 4]


    // Move n axis
    torch::Tensor a_permuted = a_cpu.permute({0, 2, 3, 1}); // shape: [1, 1, 2, 4]

    // Upload to hardware
    auto a_hw = host_to_device<int32_t>(a_permuted);
    auto p_hw = host_to_device<int32_t>(p_cpu);
    auto m_inv_hw = host_to_device<int32_t>(m_inv_cpu);
    auto perm_hw = host_to_device<int32_t>(perm_cpu);
    auto twiddles_hw = host_to_device<int32_t>(twiddles_cpu);
    auto inv_twiddles_hw = host_to_device<int32_t>(inv_twiddles_cpu);

    // Allocate result and restoration buffers
    auto result_hw = empty<int32_t>({1, 1, 2, 4});
    auto restored_hw = empty<int32_t>({1, 4, 1, 2});

    int64_t axis = -1;  // Axis of n

    // Perform NTT and inverse NTT
    ntt<int32_t, int32_t>(a_hw, p_hw, perm_hw, twiddles_hw, nullptr, nullptr, axis, false, result_hw);

    // Apply permutation to result
    torch::Tensor restored_result = device_to_host<int32_t>(result_hw);
    auto result_permuted = restored_result.permute({0, 3, 1, 2});  // shape: [1, 4, 1, 2]
    auto result_permuted_hw = host_to_device<int32_t>(result_permuted);

    intt<int32_t>(result_permuted_hw, p_hw, perm_hw, inv_twiddles_hw, m_inv_hw, nullptr, nullptr, restored_hw);

    // Download result
    torch::Tensor restored_cpu = device_to_host<int32_t>(restored_hw);

    // Assert exact match
    ASSERT_TRUE(torch::equal(restored_cpu, a_cpu))
        << "Restored input does not match the original input.\n"
        << "Expected:\n" << a_cpu << "\nActual:\n" << restored_cpu;
}

TEST(NTTTests, InplaceNTTAndINTTInt) {
    // Input tensor a: [1, 4, 1, 2]
    torch::Tensor a_cpu = torch::tensor(
        {{{{3, 7}}, {{4, 2}}, {{6, 5}}, {{9, 1}}}},
        torch::dtype(torch::kInt64)
    ); // shape: [1, 4, 1, 2]

    torch::Tensor p_cpu = torch::tensor({17, 17}, torch::dtype(torch::kInt64)); // [k=2]
    torch::Tensor m_inv_cpu = torch::tensor({13, 13}, torch::dtype(torch::kInt64)); // [k]
    torch::Tensor perm_cpu = torch::tensor({0, 2, 1, 3}, torch::dtype(torch::kInt64));
    torch::Tensor twiddles_cpu = torch::tensor({
        {1, 4, 16, 13},     // for 17
        {1, 4, 16, 13}      // for 17 again
    }, torch::dtype(torch::kInt64));  // [2, 4]
    torch::Tensor inv_twiddles_cpu = torch::tensor({
        {1, 13, 16, 4},     // for 17
        {1, 13, 16, 4}      // for 17
    }, torch::dtype(torch::kInt64));  // [2, 4]

    // Upload to hardware
    auto a_hw = host_to_device<int64_t>(a_cpu);
    auto p_hw = host_to_device<int64_t>(p_cpu);
    auto m_inv_hw = host_to_device<int64_t>(m_inv_cpu);
    auto perm_hw = host_to_device<int64_t>(perm_cpu);
    auto twiddles_hw = host_to_device<int64_t>(twiddles_cpu);
    auto inv_twiddles_hw = host_to_device<int64_t>(inv_twiddles_cpu);

    // In-place operation: result and input buffer are the same
    int64_t axis = -3;  // Axis of n

    // Forward NTT in-place
    ntt<int64_t, int64_t>(a_hw, p_hw, perm_hw, twiddles_hw, nullptr, nullptr, axis, false, a_hw);

    // Inverse NTT in-place
    intt<int64_t>(a_hw, p_hw, perm_hw, inv_twiddles_hw, m_inv_hw, nullptr, nullptr, a_hw);

    // Download result
    torch::Tensor restored_cpu = device_to_host<int64_t>(a_hw);

    // Assert exact match
    ASSERT_TRUE(torch::equal(restored_cpu, a_cpu))
        << "Restored input does not match the original input (in-place int64_t, correct NTT params).\n"
        << "Expected:\n" << a_cpu << "\nActual:\n" << restored_cpu;
}

TEST(NTTTests, WrongAxisThrows) {
    // Input tensor a: [1, 4, 1, 2] → l = 1, m = 4, r = 1, k = 2
    torch::Tensor a_cpu = torch::tensor(
        {{{{1, 2}}, {{3, 4}}, {{5, 6}}, {{7, 8}}}},
        torch::dtype(torch::kInt32)
    ); // shape: [1, 4, 1, 2]

    // Parameters
    torch::Tensor p_cpu = torch::tensor({17, 257}, torch::dtype(torch::kInt32));         // [k]
    torch::Tensor perm_cpu = torch::tensor({0, 2, 1, 3}, torch::dtype(torch::kInt32));   // [m]

    // twiddles and inv_twiddles
    torch::Tensor twiddles_cpu = torch::tensor({
        {1, 4, 2, 8},
        {1, 16, 4, 64}
    }, torch::dtype(torch::kInt32));  // [2, 4]

    // Upload to hardware
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto p_hw = host_to_device<int32_t>(p_cpu);
    auto perm_hw = host_to_device<int32_t>(perm_cpu);
    auto twiddles_hw = host_to_device<int32_t>(twiddles_cpu);
    auto result_hw = empty<int32_t>({1, 4, 1, 2});

    int64_t axis = -2;  // Wrong axis

    EXPECT_THROW((ntt<int32_t, int32_t>(a_hw, p_hw, perm_hw, twiddles_hw, nullptr, nullptr, axis, false, result_hw)), std::invalid_argument);
}
