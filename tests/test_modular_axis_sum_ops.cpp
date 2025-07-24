#include "modular_axis_sum_ops.h"
#include "device_memory.h"
#include "gtest/gtest.h"

using namespace lattica_hw_api;

TEST(AxisModSumTests, Basic3DAxis1) {
    // Input: [2, 3, 4], reduce over axis=1 → result shape [2, 4]
    torch::Tensor a = torch::tensor({
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}
    }, torch::kInt32);

    torch::Tensor p = torch::tensor({11, 13, 17, 19}, torch::kInt32);
    torch::Tensor expected = (a.sum(1)) % p;

    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({2, 4});

    axis_modsum(a_hw, p_hw, /*axis=*/1, result_hw);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    std::cout << result << std::endl;
    std::cout << expected << std::endl;
    ASSERT_TRUE(torch::equal(result, expected)) << "3D axis=1 modsum failed.";
}

TEST(AxisModSumTests, ReduceFirstAxis) {
    // Input: [3, 4], reduce over axis=0 → result shape [4]
    torch::Tensor a = torch::tensor({
        {1, 2, 3, 4},
        {4, 5, 6, 7},
        {7, 8, 9, 10}
    }, torch::kInt32);

    torch::Tensor p = torch::tensor({5, 7, 11, 13}, torch::kInt32);
    torch::Tensor expected = (a.sum(0)) % p;

    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({4});

    axis_modsum(a_hw, p_hw, /*axis=*/0, result_hw);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "2D axis=0 modsum failed.";
}

TEST(AxisModSumTests, HighDimReduction) {
    // Input: [2, 2, 2, 3], reduce over axis=2 → result shape [2, 2, 3]
    torch::Tensor a = torch::arange(24, torch::kInt32).reshape({2, 2, 2, 3});
    torch::Tensor p = torch::tensor({7, 11, 13}, torch::kInt32);
    torch::Tensor expected = (a.sum(2)) % p;

    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({2, 2, 3});

    axis_modsum(a_hw, p_hw,  /*axis=*/2, result_hw);

    torch::Tensor result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected)) << "High-dim axis=2 modsum failed.";
}

TEST(AxisModSumTests, InvalidAxisThrows) {
    torch::Tensor a = torch::randint(0, 10, {2, 2, 2}, torch::kInt32);
    torch::Tensor p = torch::tensor({7, 11}, torch::kInt32);
    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({2, 2});  // removing axis=2

    EXPECT_THROW(axis_modsum(a_hw, p_hw, -1, result_hw), std::invalid_argument);
    EXPECT_THROW(axis_modsum(a_hw, p_hw, 3, result_hw), std::invalid_argument);
}

TEST(AxisModSumTests, ModulusShapeMismatchThrows) {
    torch::Tensor a = torch::randint(0, 10, {2, 2, 2}, torch::kInt32);
    torch::Tensor p = torch::tensor({7, 11, 13}, torch::kInt32);  // invalid shape
    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = empty<int32_t>({2, 2});

    EXPECT_THROW(axis_modsum(a_hw, p_hw, 2, result_hw), std::invalid_argument);
}


/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                        MODMUL_AXIS_SUM TESTS                                   ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

/************************************************************************************************
 * modmul_axis_sum: Basic Functionality
 ***********************************************************************************************/

 TEST(ModMulAxisSumTests, BasicNoPerm_AxisMinus1) {
    // Shape: [reps, sum_size, k, n] = [2, 3, 2, 4]
    auto a = torch::tensor({
        { // rep 0
            { {1,2,3,4}, {5,6,7,8} },  // i=0
            { {9,10,11,12}, {13,14,15,16} }, // i=1
            { {17,18,19,20}, {21,22,23,24} } // i=2
        },
        { // rep 1 (same as above + 1)
            { {2,3,4,5}, {6,7,8,9} },
            { {10,11,12,13}, {14,15,16,17} },
            { {18,19,20,21}, {22,23,24,25} }
        }
    }, torch::kInt64);

    // b: [sum_size, k, n] = [3, 2, 4]
    auto b = torch::tensor({
        { {1,2,1,2}, {3,4,3,4} },
        { {5,6,5,6}, {7,8,7,8} },
        { {9,10,9,10}, {11,12,11,12} }
    }, torch::kInt64);

    // p: [k] = [13, 17]
    auto p = torch::tensor({13, 17}, torch::kInt64);

    // result: [reps, k, n] = [2, 2, 4]
    auto result = torch::zeros({2, 2, 4}, torch::kInt64);

    // call wrapper
    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = nullptr; // not used
    auto log2p_list = nullptr; // not used
    auto mu_list = nullptr; // not used
    auto result_dev = host_to_device<int64_t>(result);

    // No permutation, axis = -1
    modmul_axis_sum<int64_t>(
        a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -1, false, result_dev);

    auto result_cpu = device_to_host<int64_t>(result_dev);

    torch::Tensor expected = torch::empty({2,2,4}, torch::kInt64);

    for (int rep = 0; rep < 2; ++rep)
    for (int k = 0; k < 2; ++k)
    for (int n = 0; n < 4; ++n) {
        int64_t sum = 0;
        for (int i = 0; i < 3; ++i) {
            sum += a[rep][i][k][n].item<int64_t>() * b[i][k][n].item<int64_t>();
        }
        expected[rep][k][n] = sum % p[k].item<int64_t>();
    }

    ASSERT_TRUE(torch::allclose(result_cpu, expected));

    // Check shape
    ASSERT_EQ(result_cpu.sizes(), std::vector<int64_t>({2,2,4}));
}

TEST(ModMulAxisSumTests, BasicInt32_NoPerm_AxisMinus1) {
    // Shape: [1,2,1,3]
    auto a = torch::tensor({{
        { {1, 2, 3} },
        { {4, 5, 6} }
    }}, torch::kInt32); // [1,2,1,3]

    // b: [2,1,3]
    auto b = torch::tensor({
        { {10, 20, 30} },
        { {40, 50, 60} }
    }, torch::kInt32); // [2,1,3]

    auto p = torch::tensor({97}, torch::kInt32); // [1] (arbitrary small modulus)
    auto result = torch::zeros({1,1,3}, torch::kInt32);

    auto a_dev = host_to_device<int32_t>(a);
    auto b_dev = host_to_device<int32_t>(b);
    auto p_dev = host_to_device<int32_t>(p);
    auto perm_dev = nullptr;
    auto log2p_list = nullptr;
    auto mu_list = nullptr;
    auto result_dev = host_to_device<int32_t>(result);

    modmul_axis_sum<int32_t>(
        a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -1, false, result_dev);

    auto result_cpu = device_to_host<int32_t>(result_dev);

    // Reference computation (by hand):
    // For n=0: (1*10 + 4*40) % 97 = (10 + 160) % 97 = 170 % 97 = 73
    // For n=1: (2*20 + 5*50) % 97 = (40 + 250) % 97 = 290 % 97 = 96
    // For n=2: (3*30 + 6*60) % 97 = (90 + 360) % 97 = 450 % 97 = 62

    auto expected = torch::tensor({{{73, 96, 62}}}, torch::kInt32);

    ASSERT_TRUE(torch::equal(result_cpu, expected));
    ASSERT_EQ(result_cpu.sizes(), std::vector<int64_t>({1,1,3}));
}

TEST(ModMulAxisSumTests, Permutation_AxisMinus3) {
    // Shape: [reps, n, sum_size, k] = [1, 3, 2, 2]
    auto a = torch::arange(1, 13, torch::kInt64).reshape({1,3,2,2});
    auto b = torch::arange(101, 113, torch::kInt64).reshape({3,2,2});
    auto p = torch::tensor({7,11}, torch::kInt64);
    auto perm = torch::tensor({2,0,1}, torch::kInt64); // permutes n: 0→2, 1→0, 2→1
    auto result = torch::zeros({1,3,2}, torch::kInt64);

    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = host_to_device<int64_t>(perm);
    auto log2p_list = nullptr; // not used
    auto mu_list = nullptr; // not used
    auto result_dev = host_to_device<int64_t>(result);

    // Permutation on axis -3
    modmul_axis_sum<int64_t>(
        a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -3, true, result_dev);

    auto result_cpu = device_to_host<int64_t>(result_dev);

    // Reference calculation
    auto expected = torch::empty({1, 3, 2}, torch::kInt64);
    for (int rep = 0; rep < 1; ++rep)
    for (int n = 0; n < 3; ++n) {
        int64_t n_idx = perm[n].item<int64_t>();
        for (int k = 0; k < 2; ++k) {
            int64_t sum = 0;
            for (int i = 0; i < 2; ++i) {
                sum += a[rep][n][i][k].item<int64_t>() * b[n_idx][i][k].item<int64_t>();
            }
            // Write result at permuted location
            expected[rep][n_idx][k] = sum % p[k].item<int64_t>();
        }
    }

    ASSERT_TRUE(torch::equal(result_cpu, expected));
}

TEST(ModMulAxisSumTests, NonContiguousInputs) {
    // a_base: [2,4,2,4]
    auto a_base = torch::arange(2*4*2*4, torch::kInt64).reshape({2,4,2,4});
    // Slice dim 1: 1:3
    auto a = a_base.index({torch::indexing::Slice(), torch::indexing::Slice(1,3), torch::indexing::Slice(), torch::indexing::Slice()});
    ASSERT_FALSE(a.is_contiguous());
    // shape: [2,2,2,4]

    // b: [2,4,2] --> transpose(1,2) --> [2,2,4]
    auto b_base = torch::arange(2*4*2, torch::kInt64).reshape({2,4,2});
    auto b = b_base.transpose(1,2); // [2,2,4]
    ASSERT_FALSE(b.is_contiguous());

    auto p = torch::tensor({13, 17}, torch::kInt64);
    auto result = torch::zeros({2,2,4}, torch::kInt64);

    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = nullptr;
    auto log2p_list = nullptr;
    auto mu_list = nullptr;
    auto result_dev = host_to_device<int64_t>(result);

    modmul_axis_sum<int64_t>(
        a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -1, false, result_dev);

    auto result_cpu = device_to_host<int64_t>(result_dev);

    // Reference calculation with contiguous tensors
    torch::Tensor a_contig = a.contiguous();
    torch::Tensor b_contig = b.contiguous();
    torch::Tensor expected = torch::empty({2,2,4}, torch::kInt64);

    for (int rep = 0; rep < 2; ++rep)
    for (int k = 0; k < 2; ++k)
    for (int n = 0; n < 4; ++n) {
        int64_t sum = 0;
        for (int i = 0; i < 2; ++i) {
            sum += a_contig[rep][i][k][n].item<int64_t>() * b_contig[i][k][n].item<int64_t>();
        }
        expected[rep][k][n] = sum % p[k].item<int64_t>();
    }

    ASSERT_TRUE(torch::allclose(result_cpu, expected));
    ASSERT_EQ(result_cpu.sizes(), std::vector<int64_t>({2,2,4}));
}

TEST(ModMulAxisSumTests, HandlesInt64OverflowCorrectly) {
    // Values chosen so a * b overflows int64_t, but not uint64_t
    int64_t large1 = 0x7FFFFFFFFFFFFFFF; // INT64_MAX
    int64_t large2 = 0x7FFFFFFFFFFFFFFF; // INT64_MAX
    int64_t small_mod = 11;

    // Shape: [1,1,1,1], axis = -1
    auto a = torch::full({1,1,1,1}, large1, torch::kInt64);
    auto b = torch::full({1,1,1}, large2, torch::kInt64);
    auto p = torch::tensor({small_mod}, torch::kInt64);
    auto result = torch::zeros({1,1,1}, torch::kInt64);

    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = nullptr;
    auto log2p_list = nullptr;
    auto mu_list = nullptr;
    auto result_dev = host_to_device<int64_t>(result);

    modmul_axis_sum<int64_t>(
        a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -1, false, result_dev);

    auto result_cpu = device_to_host<int64_t>(result_dev);

    __int128 exp = (__int128(large1) * __int128(large2)) % __int128(small_mod);
    int64_t ref = static_cast<int64_t>(exp);
    ASSERT_EQ(result_cpu[0][0][0].item<int64_t>(), ref);

}

TEST(ModMulAxisSumTests, AccumulatesAcrossRepeatedCalls) {
    // Simple example: [reps, sum_size, k, n] = [1, 1, 1, 2]
    auto a1 = torch::tensor({{{{1, 2}}}}, torch::kInt64);  // [1,1,1,2]
    auto b1 = torch::tensor({{{3, 4}}}, torch::kInt64);    // [1,1,2]
    auto a2 = torch::tensor({{{{5, 6}}}}, torch::kInt64);  // [1,1,1,2]
    auto b2 = torch::tensor({{{7, 8}}}, torch::kInt64);    // [1,1,2]
    auto p = torch::tensor({13}, torch::kInt64);           // modulus [1]
    auto result = torch::zeros({1,1,2}, torch::kInt64);    // [1,1,2]

    // Device copies
    auto a1_dev = host_to_device<int64_t>(a1);
    auto b1_dev = host_to_device<int64_t>(b1);
    auto a2_dev = host_to_device<int64_t>(a2);
    auto b2_dev = host_to_device<int64_t>(b2);
    auto p_dev = host_to_device<int64_t>(p);
    std::shared_ptr<DeviceTensor<int64_t>> perm_dev = nullptr;
    std::shared_ptr<DeviceTensor<int64_t>> log2p_list = nullptr;
    std::shared_ptr<DeviceTensor<int64_t>> mu_list = nullptr;
    auto result_dev = host_to_device<int64_t>(result);

    // First call
    modmul_axis_sum<int64_t>(a1_dev, b1_dev, p_dev, perm_dev, log2p_list, mu_list, -1, false, result_dev);

    // Second call (accumulates!)
    modmul_axis_sum<int64_t>(a2_dev, b2_dev, p_dev, perm_dev, log2p_list, mu_list, -1, false, result_dev);

    // Fetch result back
    auto result_cpu = device_to_host<int64_t>(result_dev);

    // Manual reference computation (accumulation modulo 13)
    // First call:
    // [0,0,0] = (1*3) % 13 = 3
    // [0,0,1] = (2*4) % 13 = 8
    // Second call, add to previous:
    // [0,0,0] += (5*7) = 3 + 35 = 38 % 13 = 12
    // [0,0,1] += (6*8) = 8 + 48 = 56 % 13 = 4
    auto expected = torch::tensor({{{12, 4}}}, torch::kInt64);

    ASSERT_TRUE(torch::equal(result_cpu, expected));
    ASSERT_EQ(result_cpu.sizes(), std::vector<int64_t>({1,1,2}));
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

TEST(ModMulAxisSumTests, Throws_BadShape) {
    auto a = torch::empty({2,3,2,4}, torch::kInt64);
    auto b = torch::empty({2,2,4}, torch::kInt64); // bad shape
    auto p = torch::ones({2}, torch::kInt64);
    auto result = torch::zeros({2,2,4}, torch::kInt64);

    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = nullptr;
    auto log2p_list = nullptr;
    auto mu_list = nullptr;
    auto result_dev = host_to_device<int64_t>(result);

    EXPECT_THROW(
        modmul_axis_sum<int64_t>(
            a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -1, false, result_dev
        ),
        std::invalid_argument
    );
}

TEST(ModMulAxisSumTests, Throws_BadAxis) {
    auto a = torch::empty({2,3,2,4}, torch::kInt64);
    auto b = torch::empty({3,2,4}, torch::kInt64);
    auto p = torch::ones({2}, torch::kInt64);
    auto result = torch::zeros({2,2,4}, torch::kInt64);

    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = nullptr;
    auto log2p_list = nullptr;
    auto mu_list = nullptr;
    auto result_dev = host_to_device<int64_t>(result);

    EXPECT_THROW(
        modmul_axis_sum<int64_t>(
            a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, 4, false, result_dev
        ),
        std::invalid_argument
    );
}

TEST(ModMulAxisSumTests, Throws_NegativeModulus) {
    auto a = torch::empty({1,1,1,1}, torch::kInt64);
    auto b = torch::empty({1,1,1}, torch::kInt64);
    auto p = torch::tensor({-1}, torch::kInt64);
    auto result = torch::zeros({1,1,1}, torch::kInt64);

    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = std::shared_ptr<DeviceTensor<int64_t>>();
    auto log2p_list = std::shared_ptr<DeviceTensor<int64_t>>();
    auto mu_list = std::shared_ptr<DeviceTensor<int64_t>>();
    auto result_dev = host_to_device<int64_t>(result);

    EXPECT_THROW(
        modmul_axis_sum<int64_t>(
            a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -1, false, result_dev
        ),
        std::invalid_argument
    );
}

TEST(ModMulAxisSumTests, Throws_PermWrongShape) {
    // a: [1, 2, 2, 2]
    auto a = torch::ones({1,2,2,2}, torch::kInt64);
    auto b = torch::ones({2,2,2}, torch::kInt64);
    auto p = torch::ones({2}, torch::kInt64);
    auto result = torch::zeros({1,2,2}, torch::kInt64);

    // Perm tensor with wrong length (should be n=2)
    auto perm = torch::tensor({0,1,2}, torch::kInt64); // length 3, should be 2

    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = host_to_device<int64_t>(perm);
    auto log2p_list = nullptr;
    auto mu_list = nullptr;
    auto result_dev = host_to_device<int64_t>(result);

    EXPECT_THROW(
        modmul_axis_sum<int64_t>(
            a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -1, true, result_dev
        ),
        std::invalid_argument
    );
}

TEST(ModMulAxisSumTests, Throws_PermOutOfBounds) {
    // a: [1, 2, 2, 2]
    auto a = torch::ones({1,2,2,2}, torch::kInt64);
    auto b = torch::ones({2,2,2}, torch::kInt64);
    auto p = torch::ones({2}, torch::kInt64);
    auto result = torch::zeros({1,2,2}, torch::kInt64);

    // Perm tensor with out-of-bounds index
    auto perm = torch::tensor({0, 5}, torch::kInt64); // 5 >= n=2

    auto a_dev = host_to_device<int64_t>(a);
    auto b_dev = host_to_device<int64_t>(b);
    auto p_dev = host_to_device<int64_t>(p);
    auto perm_dev = host_to_device<int64_t>(perm);
    auto log2p_list = nullptr;
    auto mu_list = nullptr;
    auto result_dev = host_to_device<int64_t>(result);

    EXPECT_THROW(
        modmul_axis_sum<int64_t>(
            a_dev, b_dev, p_dev, perm_dev, log2p_list, mu_list, -1, true, result_dev
        ),
        std::invalid_argument
    );
}

