#include "gadget_decomposition_full_q.h"
#include "device_memory.h"
#include "gtest/gtest.h"

using namespace lattica_hw_api;

TEST(GadgetDecompositionFullQTests, Int64Decomposition) {

    int64_t n = 4;
    int64_t q_len = 2;

    int64_t g_exp = 16;
    int64_t g_base_bits = 4;

    torch::Tensor q_list_cpu = torch::tensor({  539754497,
                                                565374977 }, torch::dtype(torch::kInt64));

    // a of shape [1, q_len, n] (resp_l=1, q_len=2, resp_r=4) - a in CRT representation
    torch::Tensor a_cpu = torch::tensor({{{350869613,378288357,378552200,289529117},
                                          {152421091,136595599,291330274,556401325}}}, torch::dtype(torch::kInt64));

    // expected result of shape [1, g_exp, n] - gadget decomposition output
    torch::Tensor out_cpu = torch::zeros({1, g_exp, n}, torch::dtype(torch::kInt64));

    auto a_hw = host_to_device<int64_t>(a_cpu);
    auto q_list_hw = host_to_device<int64_t>(q_list_cpu);
    auto out_hw = host_to_device<int64_t>(out_cpu);

    apply_g_decomp_relative_to_full_q<int64_t>(a_hw, q_list_hw, g_exp, g_base_bits, out_hw);

    torch::Tensor result = device_to_host<int64_t>(out_hw);

    // Verify output shape
    ASSERT_EQ(result.size(0), 1);
    ASSERT_EQ(result.size(1), g_exp);
    ASSERT_EQ(result.size(2), n);

    torch::Tensor expected = torch::tensor({{{0,10,4,4},
                                            {1,6,11,9},
                                            {15,5,1,3},
                                            {14,11,11,4},
                                            {6,12,12,1},
                                            {14,12,10,15},
                                            {11,4,1,1},
                                            {6,3,4,10},
                                            {4,6,9,4},
                                            {10,13,6,0},
                                            {5,4,10,7},
                                            {6,9,8,14},
                                            {2,0,3,10},
                                            {6,15,7,8},
                                            {2,2,1,2},
                                            {0,0,0,0}}}, torch::dtype(torch::kInt64));

    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(GadgetDecompositionFullQTests, Int32Decomposition) {

    int32_t n = 2;
    int32_t q_len = 2;

    int32_t g_exp = 8;
    int32_t g_base_bits = 4;

    torch::Tensor q_list_cpu = torch::tensor({97, 101}, torch::dtype(torch::kInt32));

    // a of shape [1, q_len, n] - a in CRT representation
    torch::Tensor a_cpu = torch::tensor({{{50, 75}, {60, 80}}}, torch::dtype(torch::kInt32));

    // expected result of shape [1, g_exp, n] - gadget decomposition output
    torch::Tensor out_cpu = torch::zeros({1, g_exp, n}, torch::dtype(torch::kInt32));

    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto q_list_hw = host_to_device<int32_t>(q_list_cpu);
    auto out_hw = host_to_device<int32_t>(out_cpu);

    apply_g_decomp_relative_to_full_q<int32_t>(a_hw, q_list_hw, g_exp, g_base_bits, out_hw);

    torch::Tensor result = device_to_host<int32_t>(out_hw);

    // Verify output shape
    ASSERT_EQ(result.size(0), 1);
    ASSERT_EQ(result.size(1), g_exp);
    ASSERT_EQ(result.size(2), n);

    torch::Tensor expected = torch::tensor({{{2, 3},
                                            {6, 6},
                                            {2, 9},
                                            {1, 0},
                                            {0, 0},
                                            {0, 0},
                                            {0, 0},
                                            {0, 0}}}, torch::dtype(torch::kInt32));

    ASSERT_TRUE(torch::equal(result, expected));
}


TEST(GadgetDecompositionFullQTests, ZeroInputDecomposition) {

    int64_t n = 2;
    int64_t q_len = 2;

    int64_t g_exp = 4;
    int64_t g_base_bits = 2;

    torch::Tensor q_list_cpu = torch::tensor({5, 7}, torch::dtype(torch::kInt64));

    // Zero input
    torch::Tensor a_cpu = torch::zeros({1, q_len, n}, torch::dtype(torch::kInt64));
    torch::Tensor out_cpu = torch::zeros({1, g_exp, n}, torch::dtype(torch::kInt64));

    auto a_hw = host_to_device<int64_t>(a_cpu);
    auto q_list_hw = host_to_device<int64_t>(q_list_cpu);
    auto out_hw = host_to_device<int64_t>(out_cpu);

    apply_g_decomp_relative_to_full_q<int64_t>(a_hw, q_list_hw, g_exp, g_base_bits, out_hw);

    torch::Tensor result = device_to_host<int64_t>(out_hw);

    torch::Tensor expected_zero = torch::zeros({1, g_exp, n}, torch::dtype(torch::kInt64));
    ASSERT_TRUE(torch::equal(result, expected_zero));
}