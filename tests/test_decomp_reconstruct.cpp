#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

TEST(DecompReconstructTests, DecomposeAndReconstruct_Torch) {
    torch::Tensor a_cpu = torch::tensor({51, 29, 63}, torch::dtype(torch::kInt32)); // [3]
    int64_t power = 6;
    int64_t base_bits = 1;

    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto a_digits_hw = empty<int32_t>({3, power});

    apply_g_decomp(a_hw, a_digits_hw, power, base_bits);

    auto basis_hw = host_to_device<int32_t>(
        torch::tensor({1, 2, 4, 8, 16, 32}, torch::dtype(torch::kInt32)));
    auto p6_hw = host_to_device<int32_t>(
        torch::tensor({1024, 1024, 1024, 1024, 1024, 1024}, torch::dtype(torch::kInt32)));

    // Perform modular multiplication in-place
    modmul_ttt(a_digits_hw, basis_hw, p6_hw, a_digits_hw);

    // Reshape to [3, 6, 1] to match axis_modsum API
    a_digits_hw->reshape({3, power, 1});

    // Reconstruct
    auto a_recon_hw = empty<int32_t>({3, 1});
    auto p1_hw = host_to_device<int32_t>(torch::tensor({1024}, torch::dtype(torch::kInt32)));

    axis_modsum(a_digits_hw, p1_hw, a_recon_hw, 1);
    a_recon_hw->reshape({3});

    torch::Tensor recon_cpu = device_to_host<int32_t>(a_recon_hw);

    ASSERT_TRUE(torch::equal(recon_cpu, a_cpu)) << "Reconstruction failed.";
}
