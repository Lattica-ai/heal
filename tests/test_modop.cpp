#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

namespace expected_ops {
    static const auto modmul = [](const auto& x, const auto& y, const auto& p) {
        return (x * y) % p;
    };

    static const auto modsum = [](const auto& x, const auto& y, const auto& p) {
        return (x + y) % p;
    };
}

template <typename FExpected>
void run_modop_ttt(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& p,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, std::shared_ptr<DeviceTensor<int32_t>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw = host_to_device<int32_t>(a);
    auto b_hw = host_to_device<int32_t>(b);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = allocate_on_hardware<int32_t>(result_shape);

    kernel(a_hw, b_hw, p_hw, result_hw);
    auto result = device_to_host<int32_t>(result_hw);

    auto expected = expected_func(a, b, p);
    ASSERT_TRUE(torch::equal(result, expected)) << fail_message;
}

template <typename FExpected>
void run_modop_tensor_scalar(
    const torch::Tensor& a,
    const torch::Tensor& b,
    int32_t p_scalar,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, int32_t, std::shared_ptr<DeviceTensor<int32_t>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw = host_to_device<int32_t>(a);
    auto b_hw = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>(result_shape);

    kernel(a_hw, b_hw, p_scalar, result_hw);
    auto result = device_to_host<int32_t>(result_hw);

    auto expected = expected_func(a, b, p_scalar);
    ASSERT_TRUE(torch::equal(result, expected)) << fail_message;
}

template <typename FExpected>
void run_modop_scalar_tensor(
    const torch::Tensor& a,
    int32_t b_scalar,
    const torch::Tensor& p,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(const std::shared_ptr<DeviceTensor<int32_t>>&, int32_t, const std::shared_ptr<DeviceTensor<int32_t>>&, std::shared_ptr<DeviceTensor<int32_t>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw = host_to_device<int32_t>(a);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = allocate_on_hardware<int32_t>(result_shape);

    kernel(a_hw, b_scalar, p_hw, result_hw);
    auto result = device_to_host<int32_t>(result_hw);

    auto expected = expected_func(a, b_scalar, p);
    ASSERT_TRUE(torch::equal(result, expected)) << fail_message;
}

template <typename FExpected>
void run_modop_scalar_scalar(
    const torch::Tensor& a,
    int32_t b_scalar,
    int32_t p_scalar,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(const std::shared_ptr<DeviceTensor<int32_t>>&, int32_t, int32_t, std::shared_ptr<DeviceTensor<int32_t>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>(result_shape);

    kernel(a_hw, b_scalar, p_scalar, result_hw);
    auto result = device_to_host<int32_t>(result_hw);

    auto expected = expected_func(a, b_scalar, p_scalar);
    ASSERT_TRUE(torch::equal(result, expected)) << fail_message;
}

// ---- tcc ----
TEST(ModXXXTests, ModSumTCC) {
    auto a = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32);
    int32_t b = 5;
    int32_t p = 6;
    run_modop_scalar_scalar(a, b, p, {2, 2}, modsum_tcc<int32_t>, expected_ops::modsum, "modsum_tcc failed");
}

TEST(ModXXXTests, ModMulTCC) {
    auto a = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32);
    int32_t b = 5;
    int32_t p = 6;
    run_modop_scalar_scalar(a, b, p, {2, 2}, modmul_tcc<int32_t>, expected_ops::modmul, "modmul_tcc failed");
}

TEST(ModOpEdgeCases, ScalarBoth_TCC) {
    auto a = torch::tensor({{10, 20}}, torch::kInt32);
    int32_t b = 3;
    int32_t p = 7;
    run_modop_scalar_scalar(a, b, p, {1, 2}, modsum_tcc<int32_t>, expected_ops::modsum, "Scalar b and p in modsum_tcc failed");
}

// ---- tct ----
TEST(ModXXXTests, ModSumTCT) {
    auto a = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32);
    int32_t b = 5;
    auto p = torch::tensor({6, 7}, torch::kInt32);
    run_modop_scalar_tensor(a, b, p, {2, 2}, modsum_tct<int32_t>, expected_ops::modsum, "modsum_tct failed");
}

TEST(ModXXXTests, ModMulTCT) {
    auto a = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32);
    int32_t b = 5;
    auto p = torch::tensor({6, 7}, torch::kInt32);
    run_modop_scalar_tensor(a, b, p, {2, 2}, modmul_tct<int32_t>, expected_ops::modmul, "modmul_tct failed");
}

TEST(ModOpEdgeCases, ScalarB_TCT) {
    auto a = torch::tensor({{10, 20}}, torch::kInt32);
    int32_t b = 3;
    auto p = torch::tensor({4, 5}, torch::kInt32);
    run_modop_scalar_tensor(a, b, p, {1, 2}, modsum_tct<int32_t>, expected_ops::modsum, "Scalar b in modsum_tct failed");
}

// ---- ttc ----
TEST(ModXXXTests, ModSumTTC) {
    auto a = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32);
    auto b = torch::tensor({{5, 6}, {7, 8}}, torch::kInt32);
    int32_t p = 9;
    run_modop_tensor_scalar(a, b, p, {2, 2}, modsum_ttc<int32_t>, expected_ops::modsum, "modsum_ttc failed");
}

TEST(ModXXXTests, ModMulTTC) {
    auto a = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32);
    auto b = torch::tensor({{5, 6}, {7, 8}}, torch::kInt32);
    int32_t p = 9;
    run_modop_tensor_scalar(a, b, p, {2, 2}, modmul_ttc<int32_t>, expected_ops::modmul, "modmul_ttc failed");
}

TEST(ModOpEdgeCases, ScalarP_TTC) {
    auto a = torch::tensor({{10, 20}}, torch::kInt32);
    auto b = torch::tensor({{1, 2}}, torch::kInt32);
    int32_t p = 7;
    run_modop_tensor_scalar(a, b, p, {1, 2}, modsum_ttc<int32_t>, expected_ops::modsum, "Scalar p in modsum_ttc failed");
}


// ---- ttt ----
TEST(ModOpEdgeCases, Broadcast_TTT_3DAgainst1D) {
    auto a = torch::randint(1, 100, {2, 3, 4}, torch::kInt32);
    auto b = torch::randint(1, 100, {1, 3, 4}, torch::kInt32);
    auto p = torch::randint(100, 200, {4}, torch::kInt32);
    run_modop_ttt(a, b, p, {2, 3, 4}, modsum_ttt<int32_t>, expected_ops::modsum, "3D broadcast modsum_ttt failed");
}

TEST(ModTTTTests, BroadcastRightmost1_ModMul) {
    auto a = torch::tensor({{1}, {2}}, torch::kInt32);           // [2, 1]
    auto b = torch::tensor({10, 20, 30}, torch::kInt32);         // [3]
    auto p = torch::tensor({11, 17, 23}, torch::kInt32);         // [3]
    run_modop_ttt(a, b, p, {2, 3}, modmul_ttt<int32_t>, expected_ops::modmul, "Rightmost dim broadcast modmul failed.");
}

TEST(ModTTTTests, BroadcastRightmost1_ModSum) {
    auto a = torch::tensor({{5}, {8}}, torch::kInt32);           // [2, 1]
    auto b = torch::tensor({1, 2, 3}, torch::kInt32);            // [3]
    auto p = torch::tensor({6, 7, 8}, torch::kInt32);            // [3]
    run_modop_ttt(a, b, p, {2, 3}, modsum_ttt<int32_t>, expected_ops::modsum, "Rightmost dim broadcast modsum failed.");
}

TEST(ModOpEdgeCases, IncompatibleShapes) {
    auto a = torch::randint(0, 10, {2, 3}, torch::kInt32);
    auto b = torch::randint(0, 10, {2, 4}, torch::kInt32);
    auto p = torch::randint(100, 200, {3}, torch::kInt32);
    auto a_hw = host_to_device<int32_t>(a);
    auto b_hw = host_to_device<int32_t>(b);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = allocate_on_hardware<int32_t>({2, 3});
    EXPECT_THROW(modsum_ttt<int32_t>(a_hw, b_hw, p_hw, result_hw), std::invalid_argument);
}

TEST(ModOpEdgeCases, IncorrectPShape) {
    auto a = torch::randint(0, 10, {2, 3}, torch::kInt32);
    auto b = torch::randint(0, 10, {2, 3}, torch::kInt32);
    auto p = torch::randint(100, 200, {4, 1}, torch::kInt32);
    auto a_hw = host_to_device<int32_t>(a);
    auto b_hw = host_to_device<int32_t>(b);
    auto p_hw = host_to_device<int32_t>(p);
    auto result_hw = allocate_on_hardware<int32_t>({2, 3});
    EXPECT_THROW(modsum_ttt<int32_t>(a_hw, b_hw, p_hw, result_hw), std::invalid_argument);
}


/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                             MOD_TT TESTS                                       ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

// ──────────────────────────────────────────────────────────────────────────────
// Basic functionality
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModTTTests, BasicTensorTensor) {
    // elementwise tensor % tensor
    auto a = torch::tensor({{5, 7, 9}, {10, 12, 14}}, torch::kInt32);
    auto b = torch::tensor({{3, 5, 7}, {4, 6, 8}}, torch::kInt32);
    std::vector<int64_t> shape = {2, 3};

    auto a_hw = host_to_device<int32_t>(a);
    auto b_hw = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>(shape);

    mod_tt<int32_t>(a_hw, b_hw, result_hw);
    auto result = device_to_host<int32_t>(result_hw);

    auto expected = torch::remainder(a, b);
    ASSERT_TRUE(torch::equal(result, expected))
        << "mod_tt basic tensor-tensor remainder failed.";
}


TEST(ModTTTests, SingletonDims) {
    auto a = torch::tensor({{{7}}}, torch::kInt32);  // shape [1,1,1]
    auto b = torch::tensor({{{3}}}, torch::kInt32);
    auto result_hw = allocate_on_hardware<int32_t>({1,1,1});
    mod_tt<int32_t>(host_to_device<int32_t>(a),
                    host_to_device<int32_t>(b),
                    result_hw);
    auto out = device_to_host<int32_t>(result_hw);
    ASSERT_EQ(out.item<int32_t>(), 7 % 3);
}

// ──────────────────────────────────────────────────────────────────────────────
// High-rank and non-contiguous
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModTTTests, HighRankTensor) {
    auto a = torch::randint(1, 100, {2,2,2,2}, torch::kInt32);
    auto b = torch::randint(1, 100, {2,2,2,2}, torch::kInt32);
    auto result_hw = allocate_on_hardware<int32_t>({2,2,2,2});
    mod_tt<int32_t>(host_to_device<int32_t>(a),
                    host_to_device<int32_t>(b),
                    result_hw);
    auto out = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(out, torch::remainder(a, b)));
}

TEST(ModTTTests, NonContiguousDistinctTensors) {
    // Create a and b from different base buffers and transpose along different dims
    auto baseA = torch::arange(1, 25, torch::kInt32).reshape({2,3,4});      // [2,3,4]
    auto a     = baseA.transpose(0,2);                                      // [4,3,2], non-contiguous

    auto baseB = torch::arange(25, 49, torch::kInt32).reshape({2,3,4});     // [2,3,4]
    auto b     = baseB.permute({2,1,0});                                    // [4,3,2], non-contiguous

    std::vector<int64_t> shape = {4,3,2};
    auto a_hw     = host_to_device<int32_t>(a);
    auto b_hw     = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>(shape);

    mod_tt<int32_t>(a_hw, b_hw, result_hw);
    auto out = device_to_host<int32_t>(result_hw);

    ASSERT_TRUE(torch::equal(out, torch::remainder(a, b)))
        << "mod_tt with two distinct non-contiguous tensors failed.";
}

// ──────────────────────────────────────────────────────────────────────────────
// Error conditions
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModTTTests, NullPointerThrows) {
    auto b = torch::tensor({1,2,3}, torch::kInt32);
    auto b_hw = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>({3});
    EXPECT_THROW(mod_tt<int32_t>(nullptr, b_hw, result_hw), std::invalid_argument);
    EXPECT_THROW(mod_tt<int32_t>(b_hw, nullptr, result_hw), std::invalid_argument);
}

TEST(ModTTTests, ShapeMismatchThrows) {
    auto a = torch::randint(0, 10, {2,3}, torch::kInt32);
    auto b = torch::randint(0, 10, {2,2}, torch::kInt32);
    auto a_hw = host_to_device<int32_t>(a);
    auto b_hw = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>({2,3});
    EXPECT_THROW(mod_tt<int32_t>(a_hw, b_hw, result_hw), std::invalid_argument);
}


/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                             MOD_TC TESTS                                       ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

// ──────────────────────────────────────────────────────────────────────────────
// Basic functionality
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModTCTests, BasicTensorScalar) {
    // elementwise tensor % scalar
    auto a = torch::tensor({{5, 7, 9}, {10, 12, 14}}, torch::kInt32);
    int32_t b_scalar = 6;
    std::vector<int64_t> shape = {2, 3};

    auto a_hw = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>(shape);

    mod_tc<int32_t>(a_hw, b_scalar, result_hw);
    auto result = device_to_host<int32_t>(result_hw);

    auto expected = torch::remainder(a, b_scalar);
    ASSERT_TRUE(torch::equal(result, expected))
        << "mod_tc basic tensor-scalar remainder failed.";
}

TEST(ModTCTests, SingletonDimsScalar) {
    auto a = torch::tensor({{7}}, torch::kInt32);  // [1,1]
    auto result_hw = allocate_on_hardware<int32_t>({1,1});
    mod_tc<int32_t>(host_to_device<int32_t>(a), 3, result_hw);
    auto out = device_to_host<int32_t>(result_hw);
    ASSERT_EQ(out.item<int32_t>(), 7 % 3);
}

// ──────────────────────────────────────────────────────────────────────────────
// High-rank and non-contiguous
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModTCTests, HighRankTensorScalar) {
    auto a = torch::randint(1, 50, {2,2,3,2}, torch::kInt32);
    auto result_hw = allocate_on_hardware<int32_t>({2,2,3,2});
    mod_tc<int32_t>(host_to_device<int32_t>(a), 7, result_hw);
    auto out = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(out, torch::remainder(a, 7)));
}

TEST(ModTCTests, NonContiguousTensorScalar) {
    auto base = torch::arange(1, 13, torch::kInt32).reshape({2,2,3});
    auto a = base.transpose(1,2);  // [2,3,2]
    auto result_hw = allocate_on_hardware<int32_t>({2,3,2});
    mod_tc<int32_t>(host_to_device<int32_t>(a), 4, result_hw);
    auto out = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(out, torch::remainder(a, 4)));
}

// ──────────────────────────────────────────────────────────────────────────────
// Error conditions
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModTCTests, NullPointerThrows) {
    auto a = torch::tensor({1,2,3}, torch::kInt32);
    auto a_hw = host_to_device<int32_t>(a);
    auto result_hw = allocate_on_hardware<int32_t>({3});
    EXPECT_THROW(mod_tc<int32_t>(nullptr, 5, result_hw), std::invalid_argument);
}

TEST(ModTCTests, ShapeMismatchThrows) {
    auto a = torch::randint(0, 10, {2,3}, torch::kInt32);
    auto a_hw = host_to_device<int32_t>(a);
    // wrong result shape
    auto result_hw = allocate_on_hardware<int32_t>({3,2});
    EXPECT_THROW(mod_tc<int32_t>(a_hw, 5, result_hw), std::invalid_argument);
}


/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                             MOD_CT TESTS                                       ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

// ──────────────────────────────────────────────────────────────────────────────
// Basic functionality
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModCTTests, BasicScalarTensor) {
    // elementwise scalar % tensor
    int32_t a_scalar = 6;
    auto b = torch::tensor({{3, 5, 7}, {4, 6, 8}}, torch::kInt32);
    std::vector<int64_t> shape = {2, 3};

    auto b_hw = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>(shape);

    mod_ct<int32_t>(a_scalar, b_hw, result_hw);
    auto result = device_to_host<int32_t>(result_hw);

    auto full_a = torch::full(b.sizes(), a_scalar, torch::kInt32);
    auto expected = torch::remainder(full_a, b);
    ASSERT_TRUE(torch::equal(result, expected))
        << "mod_ct basic scalar-tensor remainder failed.";
}

TEST(ModCTTests, ZeroNumerator) {
    // edge: zero numerator
    int32_t a_scalar = 0;
    auto b = torch::tensor({{1, 2}, {3, 4}}, torch::kInt32);
    std::vector<int64_t> shape = {2, 2};

    auto b_hw = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>(shape);

    mod_ct<int32_t>(a_scalar, b_hw, result_hw);
    auto result = device_to_host<int32_t>(result_hw);

    auto full_a = torch::full(b.sizes(), a_scalar, torch::kInt32);
    auto expected = torch::remainder(full_a, b);
    ASSERT_TRUE(torch::equal(result, expected))
        << "mod_ct zero-numerator failed.";
}

TEST(ModCTTests, SingletonDimsScalarTensor) {
    auto b = torch::tensor({{3}}, torch::kInt32);  // [1,1]
    auto result_hw = allocate_on_hardware<int32_t>({1,1});
    mod_ct<int32_t>(8, host_to_device<int32_t>(b), result_hw);
    auto out = device_to_host<int32_t>(result_hw);
    auto full_a = torch::full(b.sizes(), 8, torch::kInt32);
    ASSERT_EQ(out.item<int32_t>(), (full_a % b).item<int32_t>());
}

// ──────────────────────────────────────────────────────────────────────────────
// High-rank and non-contiguous
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModCTTests, HighRankScalarTensor) {
    auto b = torch::randint(1, 20, {2,3,2,2}, torch::kInt32);
    auto result_hw = allocate_on_hardware<int32_t>({2,3,2,2});
    mod_ct<int32_t>(9, host_to_device<int32_t>(b), result_hw);
    auto out = device_to_host<int32_t>(result_hw);
    auto full_a = torch::full(b.sizes(), 9, torch::kInt32);
    ASSERT_TRUE(torch::equal(out, torch::remainder(full_a, b)));
}

TEST(ModCTTests, NonContiguousScalarTensor) {
    auto base = torch::arange(1, 19, torch::kInt32).reshape({2,3,3});  // [2,3,3]
    auto b = base.transpose(0,2);                                // now [3,3,2]
    auto result_hw = allocate_on_hardware<int32_t>({3,3,2});
    mod_ct<int32_t>(4, host_to_device<int32_t>(b), result_hw);
    auto out = device_to_host<int32_t>(result_hw);
    auto full_a = torch::full(b.sizes(), 4, torch::kInt32);
    ASSERT_TRUE(torch::equal(out, torch::remainder(full_a, b)));
}


// ──────────────────────────────────────────────────────────────────────────────
// Error conditions
// ──────────────────────────────────────────────────────────────────────────────

TEST(ModCTTests, NullPointerThrows) {
    auto b = torch::tensor({1,2,3}, torch::kInt32);
    auto b_hw = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>({3});
    EXPECT_THROW(mod_ct<int32_t>(5, nullptr, result_hw), std::invalid_argument);
}

TEST(ModCTTests, ShapeMismatchThrows) {
    auto b = torch::randint(0, 10, {2,3}, torch::kInt32);
    auto b_hw = host_to_device<int32_t>(b);
    auto result_hw = allocate_on_hardware<int32_t>({3,2});     // wrong result shape
    EXPECT_THROW(mod_ct<int32_t>(5, b_hw, result_hw), std::invalid_argument);
}
