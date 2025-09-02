#include "gadget_decomposition_full_q.h"
#include "device_memory.h"
#include "tensor_layout_ops.h"
#include "modular_axis_sum_ops.h"
#include "modular_ops.h"
#include "gtest/gtest.h"

using namespace lattica_hw_api;

TEST(DecompReconstructTests, DecomposeAndReconstruct_FullQ) {

    int64_t reps_l = 1, reps_r = 2;
    int64_t g_exp = 6;
    int64_t g_base_bits = 3;
    int64_t g_base = 8;

    torch::Tensor q_list_cpu = torch::tensor({97, 101, 103}, torch::dtype(torch::kInt64));
    torch::Tensor q_cpu = torch::tensor({1009091}, torch::dtype(torch::kInt64));

    // Test values: 12345 and 67890
    // a of shape [reps_l, q_len, reps_r] - a in CRT representation
    torch::Tensor a_cpu = torch::tensor({{{26, 87}, {23, 18}, {88, 13}}}, torch::dtype(torch::kInt64));
    torch::Tensor expected_cpu = torch::tensor({{12345, 67890}}, torch::dtype(torch::kInt64));
    torch::Tensor out_cpu = torch::zeros({reps_l, g_exp, reps_r}, torch::dtype(torch::kInt64));

    // Create device tensors
    auto a_hw = host_to_device<int64_t>(a_cpu);
    auto q_list_hw = host_to_device<int64_t>(q_list_cpu);
    auto out_hw = host_to_device<int64_t>(out_cpu);
    auto modulus_hw = host_to_device<int64_t>(q_cpu);

    // Apply gadget decomposition
    apply_g_decomp_relative_to_full_q<int64_t>(a_hw, q_list_hw, g_exp, g_base_bits, out_hw);

    auto basis_hw = host_to_device<int64_t>(
        torch::tensor({{1}, {8}, {64}, {512}, {4096}, {32768}}, torch::dtype(torch::kInt64)));

    // Perform modular multiplication in place - out_hw of shape [reps_l, g_exp, reps_r]
    modmul_ttt(out_hw, basis_hw, modulus_hw, out_hw);

    // Reconstruct by summing along the g_exp axis (axis=1)
    auto reconstructed_hw = empty<int64_t>({reps_l, reps_r, 1});
    out_hw = reshape(out_hw, {reps_l, g_exp, reps_r, 1});
    axis_modsum(out_hw, modulus_hw, 1, reconstructed_hw);
    reconstructed_hw = reshape(reconstructed_hw, {reps_l, reps_r});
    torch::Tensor reconstructed_cpu = device_to_host<int64_t>(reconstructed_hw);

    ASSERT_TRUE(torch::equal(reconstructed_cpu, expected_cpu));
}
