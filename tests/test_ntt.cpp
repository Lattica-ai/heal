#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

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
    ntt<int32_t>(a_hw, p_hw, perm_hw, twiddles_hw, nullptr, nullptr, result_hw, axis);

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
