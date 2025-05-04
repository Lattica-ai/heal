#include "gtest/gtest.h"
#include "test_helpers.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                           AXIS_MODSUM TESTS                                    ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality (int32)
 ****************************************************************************************/

 TEST(AxisModSumTests, Basic3DAxis1_Int32) {
    // Input: [2, 3, 4], reduce over axis=1 → result shape [2, 4]
    torch::Tensor a = torch::tensor({
        {{ 1,  2,  3,  4},
         { 5,  6,  7,  8},
         { 9, 10, 11, 12}},
        {{13, 14, 15, 16},
         {17, 18, 19, 20},
         {21, 22, 23, 24}}
    }, torch::kInt32);
    torch::Tensor p = torch::tensor({11, 13, 17, 19}, torch::kInt32);
    torch::Tensor expected = (a.sum(1)) % p;

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({2, 4});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/1);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(AxisModSumTests, ReduceFirstAxis_Int32) {
    // Input: [3, 4], reduce over axis=0 → result shape [4]
    torch::Tensor a = torch::tensor({
        {1, 2, 3, 4},
        {4, 5, 6, 7},
        {7, 8, 9,10}
    }, torch::kInt32);
    torch::Tensor p = torch::tensor({5, 7, 11, 13}, torch::kInt32);
    torch::Tensor expected = (a.sum(0)) % p;

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({4});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/0);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(AxisModSumTests, HighDimReduction_Int32) {
    // Input: [2, 2, 2, 3], reduce over axis=2 → result shape [2, 2, 3]
    torch::Tensor a = torch::arange(24, torch::kInt32).reshape({2,2,2,3});
    torch::Tensor p = torch::tensor({7, 11, 13}, torch::kInt32);
    torch::Tensor expected = (a.sum(2)) % p;

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({2,2,3});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/2);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(AxisModSumTests, Basic3DAxis0_Int32) {
    // Input: [2, 3, 4], reduce over axis=0 → result shape [3, 4]
    torch::Tensor a = torch::tensor({
        {{ 1,  2,  3,  4},
         { 5,  6,  7,  8},
         { 9, 10, 11, 12}},
        {{13, 14, 15, 16},
         {17, 18, 19, 20},
         {21, 22, 23, 24}}
    }, torch::kInt32);
    torch::Tensor p = torch::tensor({17, 19, 23, 29}, torch::kInt32);
    torch::Tensor expected = (a.sum(0)) % p;

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({3,4});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/0);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

/************************************************************************************************
 * Basic functionality (int64)
 ****************************************************************************************/

 TEST(AxisModSumTests, Basic3DAxis1_Int64) {
    // shape [2,2,3], dtype=int64
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt64);

    // plain '1,2,3' etc. — the dtype comes from opts
    auto a = torch::tensor({
        {{ 1,  2,  3}, { 4,  5,  6}},
        {{ 7,  8,  9}, {10, 11, 12}}
    }, opts);

    auto p = torch::tensor({5, 7, 11}, opts);

    auto expected = (a.sum(1)) % p;  // shape [2,3]

    auto a_hw      = host_to_device<int64_t>(a);
    auto p_hw      = host_to_device<int64_t>(p);
    auto result_hw = zeros<int64_t>({2,3});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/1);

    auto result = device_to_host<int64_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

/************************************************************************************************
 * More AxisModSum edge & corner-case tests
 ***********************************************************************************************/


// sum_size = 1 (unit-length reduction)
TEST(AxisModSumTests, SumSizeOne_Int32) {
    // a: [2,3,1], p: [5]
    // reduce over axis=1 → result shape [2,1]
    torch::Tensor a = torch::tensor({
        {{  7}, { 12}, { 18}},
        {{ -3}, {  0}, {  4}}
    }, torch::kInt32);            // shape [2,3,1]
    torch::Tensor p = torch::tensor({5}, torch::kInt32);

    // sum over axis=1, then %5 → shape [2,1]
    auto expected = a.sum(1).fmod(p);

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({2,1});

    // now axis=1 is valid (ndim=3 → axis∈[0,1])
    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/1);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}


// Single-channel (k=1)
TEST(AxisModSumTests, SingleChannel_Int32) {
    // a: [2,4,1], p=[7]
    torch::Tensor a = torch::randint(-10, 10, {2,4,1}, torch::kInt32);
    torch::Tensor p = torch::tensor({7},             torch::kInt32);

    auto expected = a.sum(1).remainder(p);  // shape [2,1]

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({2,1});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/1);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

// * Negative values in a, non-negative result
TEST(AxisModSumTests, NegativeInputs_Int32) {
    // a has negatives, ensure (sum mod p) ∈ [0,p)
    torch::Tensor a = torch::tensor({
        {{ -3,  4, -7},
         {  5, -6,  8}}
    }, torch::kInt32);  // [1,2,3]
    torch::Tensor p = torch::tensor({5,7,11}, torch::kInt32);
    // expected uses remainder() to get non-negative
    auto expected = a.sum(1).remainder(p);  // shape [1,3]

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({1,3});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/1);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

// High-dim broadcast
TEST(AxisModSumTests, HighDimBatch_Int32) {
    // a: [2,2,3,4], reduce axis=2 → result [2,2,4]
    torch::Tensor a = torch::arange(48, torch::kInt32).reshape({2,2,3,4});
    torch::Tensor p = torch::tensor({7,11,13,17}, torch::kInt32);

    auto expected = a.sum(2).remainder(p);

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({2,2,4});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/2);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

// sum over first axis with 1D result
TEST(AxisModSumTests, SumFirstAxisTo1D_Int32) {
    // a: [5,4], reduce axis=0 → result [4]
    torch::Tensor a = torch::randint(0, 10, {5,4}, torch::kInt32);
    torch::Tensor p = torch::tensor({3,5,7,11}, torch::kInt32);

    auto expected = a.sum(0).remainder(p);

    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({4});

    axis_modsum(a_hw, p_hw, result_hw, /*axis=*/0);

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

 VALIDATION_TEST(AxisModSumTests, InvalidAxis_Throws) {
    torch::Tensor a = torch::randint(0, 10, {2,2,2}, torch::kInt32);
    torch::Tensor p = torch::tensor({7,11}, torch::kInt32);
    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({2,2});  // would be correct shape for axis=2, but axis=2 is invalid

    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, /*axis=*/-1), std::invalid_argument);
    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, /*axis=*/ 2), std::invalid_argument);
}

VALIDATION_TEST(AxisModSumTests, ModulusShapeMismatch_Throws) {
    // last dim of a is 2, but p has length 3 → mismatch
    torch::Tensor a = torch::randint(0, 10, {2,2,2}, torch::kInt32);
    torch::Tensor p = torch::tensor({5,7,11}, torch::kInt32);
    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({2,2});

    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, /*axis=*/2), std::invalid_argument);
}

VALIDATION_TEST(AxisModSumTests, PNot1D_Throws) {
    // p must be 1D
    torch::Tensor a = torch::randint(0, 10, {4,5}, torch::kInt32);
    torch::Tensor p = torch::randint(1, 10, {2,3}, torch::kInt32);  // 2D
    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({4,5});

    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, /*axis=*/0), std::invalid_argument);
}

VALIDATION_TEST(AxisModSumTests, ResultShapeMismatch_Throws) {
    // correct would be [3,4] for axis=0, but we give [4,3]
    torch::Tensor a = torch::randint(0, 10, {3,4}, torch::kInt32);
    torch::Tensor p = torch::randint(1, 10, {4},   torch::kInt32);
    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({4,3});  // wrong

    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, /*axis=*/0), std::invalid_argument);
}

VALIDATION_TEST(AxisModSumTests, InvalidPValue_Throws) {
    torch::Tensor a = torch::randint(0, 10, {2,2,2}, torch::kInt32);
    torch::Tensor p = torch::tensor({7,-11}, torch::kInt32);
    auto a_hw      = host_to_device<int32_t>(a);
    auto p_hw      = host_to_device<int32_t>(p);
    auto result_hw = zeros<int32_t>({2,2});  // would be correct shape for axis=2, but axis=2 is invalid

    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, /*axis=*/-1), std::invalid_argument);
    EXPECT_THROW(axis_modsum(a_hw, p_hw, result_hw, /*axis=*/ 2), std::invalid_argument);
}





/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                       MODMUL_AXIS_SUM TESTS                                    ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality (int32)
 ****************************************************************************************/

 TEST(ModMulAxisSumTests, BasicNoPerm_Int32) {
    // a: [2,4,3,2], b: [4,3,2], p: [7,11], no perm
    torch::Tensor a = torch::tensor({
        {{{1,2},{3,4},{5,6}},
         {{7,8},{9,10},{11,12}},
         {{13,14},{15,16},{17,18}},
         {{19,20},{21,22},{23,24}}},
        {{{25,26},{27,28},{29,30}},
         {{31,32},{33,34},{35,36}},
         {{37,38},{39,40},{41,42}},
         {{43,44},{45,46},{47,48}}}
    }, torch::kInt32);  // shape [2,4,3,2]

    torch::Tensor b = torch::tensor({
        {{2,3},{4,5},{6,7}},
        {{8, 9},{10,11},{12,13}},
        {{14,15},{16,17},{18,19}},
        {{20,21},{22,23},{24,25}}
    }, torch::kInt32); // [4,3,2]

    torch::Tensor p = torch::tensor({7,11}, torch::kInt32);

    // compute expected on CPU:
    // broadcast b to [2,4,3,2], multiply, sum over dim=2, then mod p
    auto b_expanded = b.unsqueeze(0).expand({2,4,3,2});
    auto expected = (a * b_expanded).sum(2).fmod(p);

    auto a_hw      = host_to_device<int32_t>(a);
    auto b_hw      = host_to_device<int32_t>(b);
    auto p_hw      = host_to_device<int32_t>(p);
    auto perm_hw   = host_to_device<int64_t>(torch::tensor({0,1,2,3}, torch::kInt64));
    auto result_hw = zeros<int32_t>({2,4,2});

    // apply_perm=false should ignore perm
    modmul_axis_sum(a_hw, b_hw, p_hw, perm_hw, result_hw, /*apply_perm=*/false);
    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(ModMulAxisSumTests, BasicWithPerm_Int32) {
    // a: [2,4,3,2], b: [4,3,2], p: [7,11], no perm
    torch::Tensor a = torch::tensor({
        {{{1,2},{3,4},{5,6}},
            {{7,8},{9,10},{11,12}},
            {{13,14},{15,16},{17,18}},
            {{19,20},{21,22},{23,24}}},
        {{{25,26},{27,28},{29,30}},
            {{31,32},{33,34},{35,36}},
            {{37,38},{39,40},{41,42}},
            {{43,44},{45,46},{47,48}}}
    }, torch::kInt32);  // shape [2,4,3,2]

    torch::Tensor b = torch::tensor({
        {{2,3},{4,5},{6,7}},
        {{8, 9},{10,11},{12,13}},
        {{14,15},{16,17},{18,19}},
        {{20,21},{22,23},{24,25}}
    }, torch::kInt32); // [4,3,2]

    torch::Tensor p = torch::tensor({7,11}, torch::kInt32);
    torch::Tensor perm = torch::tensor({2,0,3,1}, torch::kInt64);
    // permute b along dim 0
    auto b_perm = b.index_select(0, perm);
    auto b_perm_exp = b_perm.unsqueeze(0).expand({2,4,3,2});
    auto expected_perm = (a * b_perm_exp).sum(2).fmod(p);

    auto a_hw      = host_to_device<int32_t>(a);
    auto b_hw      = host_to_device<int32_t>(b);
    auto p_hw      = host_to_device<int32_t>(p);
    auto perm_hw   = host_to_device<int64_t>(perm);
    auto result_hw = zeros<int32_t>({2,4,2});
    modmul_axis_sum(a_hw, b_hw, p_hw, perm_hw, result_hw, /*apply_perm=*/true);

    auto result2 = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result2, expected_perm));
}

/************************************************************************************************
 * Basic functionality (int64)
 ****************************************************************************************/

 TEST(ModMulAxisSumTests, Basic4D_Int64_NoPerm) {
    // reps=1, n=2, sum_size=3, k=3
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt64);

    // a: [1,2,3,3]
    auto a = torch::tensor({
        {   // rep = 0
            {{ 1,  2,  3},
             { 4,  5,  6},
             { 7,  8,  9}},
            {{10, 11, 12},
             {13, 14, 15},
             {16, 17, 18}}
        }
    }, opts);

    // b: [2,3,3]  (n=2, sum_size=3, k=3)
    auto b = torch::tensor({
        {{ 2,  3,  5},
         { 7, 11, 13},
         {17, 19, 23}},
        {{29, 31, 37},
         {41, 43, 47},
         {53, 59, 61}}
    }, opts);

    // per-channel modulus k=3
    auto p = torch::tensor({5, 7, 11}, opts);

    // identity perm (not used here)
    auto perm = torch::tensor({0, 1}, torch::kInt64);

    // Compute expected on CPU: broadcast b to [1,2,3,3], multiply, sum over dim=2, then mod p
    auto b_expanded = b.unsqueeze(0);                // [1,2,3,3]
    auto expected   = (a * b_expanded).sum(2).fmod(p); // [1,2,3]

    // Move to device
    auto a_hw    = host_to_device<int64_t>(a);
    auto b_hw    = host_to_device<int64_t>(b);
    auto p_hw    = host_to_device<int64_t>(p);
    auto perm_hw = host_to_device<int64_t>(perm);
    auto res_hw  = zeros<int64_t>({1,2,3});

    // apply_perm=false
    modmul_axis_sum(a_hw, b_hw, p_hw, perm_hw, res_hw, /*apply_perm=*/false);

    auto result = device_to_host<int64_t>(res_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

/************************************************************************************************
 * Edge & corner‐case tests for modmul_axis_sum
 ***********************************************************************************************/

 // sum_size = 1  (unit‐length dot‐product)
TEST(ModMulAxisSumTests, SumSizeOne_Int32) {
    // a: [2,3,1,4], b: [3,1,4], p: [13,17,19,23]
    torch::Tensor a = torch::tensor({
        {{{ 1,  2,  3,  4}},
         {{ 5,  6,  7,  8}},
         {{ 9, 10, 11, 12}}},
        {{{13, 14, 15, 16}},
         {{17, 18, 19, 20}},
         {{21, 22, 23, 24}}}
    }, torch::kInt32);  // [2,3,1,4]

    torch::Tensor b = torch::tensor({
        {{2,3,5,7}},
        {{11,13,17,19}},
        {{23,29,31,37}}
    }, torch::kInt32); // [3,1,4]

    torch::Tensor p = torch::tensor({13,17,19,23}, torch::kInt32);

    // broadcast b → [2,3,1,4], mul, sum over dim=2 (trivial), mod p
    auto b_exp = b.unsqueeze(0).expand({2,3,1,4});
    auto expected = (a * b_exp).sum(2).fmod(p);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(torch::tensor({0,1,2}, torch::kInt64));
    auto res_hw  = zeros<int32_t>({2,3,4});

    // apply_perm = false (perm ignored)
    modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,false);
    auto result = device_to_host<int32_t>(res_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

 // k = 1 (single channel)
TEST(ModMulAxisSumTests, SingleChannel_Int32) {
    // a: [2,3,4,1], b: [3,4,1], p=[5]
    torch::Tensor a = torch::randint(-5, 6, {2,3,4,1}, torch::kInt32);
    torch::Tensor b = torch::randint(-5, 6, {3,4,1},   torch::kInt32);
    torch::Tensor p = torch::tensor({5},              torch::kInt32);

    auto b_exp = b.unsqueeze(0).expand({2,3,4,1});
    auto expected = (a * b_exp).sum(2).remainder(p);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(torch::tensor({0,1,2}, torch::kInt64)); // ignored
    auto res_hw  = zeros<int32_t>({2,3,1});

    modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,false);
    auto result = device_to_host<int32_t>(res_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}


 // broadcast b across reps
TEST(ModMulAxisSumTests, BroadcastBAcrossReps_Int32) {
    // a: [3,2,2,2], b: [2,2,2]
    torch::Tensor a = torch::randint(0,10,{3,2,2,2}, torch::kInt32);
    torch::Tensor b = torch::randint(0,10,{2,2,2},   torch::kInt32);
    torch::Tensor p = torch::tensor({5,7},           torch::kInt32);

    auto b_exp = b.unsqueeze(0).expand({3,2,2,2});
    auto expected = (a * b_exp).sum(2).fmod(p);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(torch::tensor({0,1}, torch::kInt64));
    auto res_hw  = zeros<int32_t>({3,2,2});

    modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,false);
    auto result = device_to_host<int32_t>(res_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

// empty batch (reps=0)
TEST(ModMulAxisSumTests, EmptyBatch_Int32) {
    // a: [0,3,2,2], b: [3,2,2], p: [5,7]
    torch::Tensor a = torch::randint(0,10,{0,3,2,2}, torch::kInt32);
    torch::Tensor b = torch::randint(0,10,{3,2,2},   torch::kInt32);
    torch::Tensor p = torch::tensor({5,7},           torch::kInt32);

    auto expected = torch::zeros({0,3,2}, torch::kInt32);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(torch::tensor({0,1,2}, torch::kInt64));
    auto res_hw  = zeros<int32_t>({0,3,2});

    modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,false);
    auto result = device_to_host<int32_t>(res_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

/************************************************************************************************
 * Error conditions
 ****************************************************************************************/

VALIDATION_TEST(ModMulAxisSumTests, PNot1D_Throws) {
    // p must be 1D
    auto a = torch::randint(0,10,{2,3,4}, torch::kInt32);
    auto b = torch::randint(0,10,{3,4},   torch::kInt32);
    auto p = torch::randint(1,10,{2,3},   torch::kInt32);
    auto perm = torch::tensor({0,1}, torch::kInt64);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(perm);
    auto res_hw  = zeros<int32_t>({2,3,4});

    EXPECT_THROW(modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,true), std::invalid_argument);
}

VALIDATION_TEST(ModMulAxisSumTests, LastDimMismatchA_Throws) {
    // last dim of a != p length
    auto a = torch::randint(0,10,{2,3,5}, torch::kInt32);
    auto b = torch::randint(0,10,{3,5},   torch::kInt32);
    auto p = torch::tensor({7,11,13},     torch::kInt32);  // length=3
    auto perm = torch::tensor({0,1}, torch::kInt64);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(perm);
    auto res_hw  = zeros<int32_t>({2,3,3});

    EXPECT_THROW(modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,false), std::invalid_argument);
}

VALIDATION_TEST(ModMulAxisSumTests, BShapeMismatch_Throws) {
    // b must match a in all dims except sum_axis
    auto a = torch::randint(0,10,{2,3,4,5}, torch::kInt32);
    auto b = torch::randint(0,10,{2,4,5},   torch::kInt32); // wrong leading dim
    auto p = torch::tensor({2,3,5,7,11},     torch::kInt32);
    auto perm = torch::tensor({0,1}, torch::kInt64);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(perm);
    auto res_hw  = zeros<int32_t>({2,3,5});

    EXPECT_THROW(modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,false), std::invalid_argument);
}

VALIDATION_TEST(ModMulAxisSumTests, PermShapeMismatch_Throws) {
    // apply_perm=true but perm length != size of axis before sum_axis
    auto a = torch::randint(0,10,{2,3,4,5}, torch::kInt32);
    auto b = torch::randint(0,10,{3,4,5},   torch::kInt32);
    auto p = torch::tensor({2,3,5,7,11},     torch::kInt32);
    auto perm = torch::tensor({0,1}, torch::kInt64); // length=2 but needs to be 3

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(perm);
    auto res_hw  = zeros<int32_t>({2,3,5});

    EXPECT_THROW(modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,true), std::invalid_argument);
}

VALIDATION_TEST(ModMulAxisSumTests, ResultShapeMismatch_Throws) {
    // result must have same dims as a except sum_axis removed
    auto a = torch::randint(0,10,{2,3,4,5}, torch::kInt32);
    auto b = torch::randint(0,10,{3,4,5},   torch::kInt32);
    auto p = torch::tensor({2,3,5,7,11},     torch::kInt32);
    auto perm = torch::tensor({0,1,2}, torch::kInt64);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(perm);
    // wrong: here sum_axis=2 removed dims so expected [2,3,5], but we give [2,4,5]
    auto res_hw  = zeros<int32_t>({2,4,5});

    EXPECT_THROW(modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,false), std::invalid_argument);
}

VALIDATION_TEST(ModMulAxisSumTests, InvalidPValue_Throws) {
    // negative or zero modulus should throw
    auto a = torch::randint(0,10,{2,3,4}, torch::kInt32);
    auto b = torch::randint(0,10,{3,4},   torch::kInt32);
    auto p = torch::tensor({5,0,7},       torch::kInt32);
    auto perm = torch::tensor({0,1,2}, torch::kInt64);

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(perm);
    auto res_hw  = zeros<int32_t>({2,3,7}); // last dim=7 to match p.size()

    EXPECT_THROW(modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,false), std::invalid_argument);
}

// perm index out‐of‐bounds
VALIDATION_TEST(ModMulAxisSumTests, PermIndexOutOfBounds_Throws) {
    // perm contains an index equal to axis length (2), which is invalid
    auto a    = torch::randint(0,10,{1,2,2,2}, torch::kInt32);
    auto b    = torch::randint(0,10,{2,2,2},    torch::kInt32);
    auto p    = torch::tensor({5,7},            torch::kInt32);
    auto perm = torch::tensor({0,2},            torch::kInt64);  // index 2 is out‐of‐bounds for size=2

    auto a_hw    = host_to_device<int32_t>(a);
    auto b_hw    = host_to_device<int32_t>(b);
    auto p_hw    = host_to_device<int32_t>(p);
    auto perm_hw = host_to_device<int64_t>(perm);
    auto res_hw  = zeros<int32_t>({1,2,2});

    EXPECT_THROW(modmul_axis_sum(a_hw,b_hw,p_hw,perm_hw,res_hw,true), std::invalid_argument);
}