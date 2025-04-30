#include "gtest/gtest.h"
#include "test_helpers.h"
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

    static const auto modneg = [](const auto& x, const auto& p) {
        return (-x) % p;
    };
}

template <typename T, typename FExpected>
void run_modop_ttt(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& p,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(
      const std::shared_ptr<DeviceTensor<T>>&,
      const std::shared_ptr<DeviceTensor<T>>&,
      const std::shared_ptr<DeviceTensor<T>>&,
      std::shared_ptr<DeviceTensor<T>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw      = host_to_device<T>(a);
    auto b_hw      = host_to_device<T>(b);
    auto p_hw      = host_to_device<T>(p);
    auto result_hw = zeros<T>(result_shape);

    kernel(a_hw, b_hw, p_hw, result_hw);
    auto result   = device_to_host<T>(result_hw);
    auto expected = expected_func(a, b, p);

    ASSERT_TRUE(torch::equal(result, expected)) << fail_message;
}

template <typename T, typename FExpected>
void run_modop_tensor_scalar(
    const torch::Tensor& a,
    const torch::Tensor& b,
    T p_scalar,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(
      const std::shared_ptr<DeviceTensor<T>>&,
      const std::shared_ptr<DeviceTensor<T>>&,
      T,
      std::shared_ptr<DeviceTensor<T>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw      = host_to_device<T>(a);
    auto b_hw      = host_to_device<T>(b);
    auto result_hw = zeros<T>(result_shape);

    kernel(a_hw, b_hw, p_scalar, result_hw);
    auto result   = device_to_host<T>(result_hw);
    auto expected = expected_func(a, b, p_scalar);

    ASSERT_TRUE(torch::equal(result, expected)) << fail_message;
}

template <typename T, typename FExpected>
void run_modop_scalar_tensor(
    const torch::Tensor& a,
    T b_scalar,
    const torch::Tensor& p,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(
      const std::shared_ptr<DeviceTensor<T>>&,
      T,
      const std::shared_ptr<DeviceTensor<T>>&,
      std::shared_ptr<DeviceTensor<T>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw      = host_to_device<T>(a);
    auto p_hw      = host_to_device<T>(p);
    auto result_hw = zeros<T>(result_shape);

    kernel(a_hw, b_scalar, p_hw, result_hw);
    auto result   = device_to_host<T>(result_hw);
    auto expected = expected_func(a, b_scalar, p);

    ASSERT_TRUE(torch::equal(result, expected)) << fail_message;
}

template <typename T, typename FExpected>
void run_modop_scalar_scalar(
    const torch::Tensor& a,
    T b_scalar,
    T p_scalar,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(
      const std::shared_ptr<DeviceTensor<T>>&,
      T,
      T,
      std::shared_ptr<DeviceTensor<T>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw      = host_to_device<T>(a);
    auto result_hw = zeros<T>(result_shape);

    kernel(a_hw, b_scalar, p_scalar, result_hw);
    auto result   = device_to_host<T>(result_hw);
    auto expected = expected_func(a, b_scalar, p_scalar);

    ASSERT_TRUE(torch::equal(result, expected)) << fail_message;
}


template <typename T, typename FExpected>
void run_modneg_tt(
    const torch::Tensor& a,
    const torch::Tensor& p,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(
        const std::shared_ptr<DeviceTensor<T>>&,
        const std::shared_ptr<DeviceTensor<T>>&,
        std::shared_ptr<DeviceTensor<T>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw      = host_to_device<T>(a);
    auto p_hw      = host_to_device<T>(p);
    auto result_hw = zeros<T>(result_shape);

    kernel(a_hw, p_hw, result_hw);
    auto result   = device_to_host<T>(result_hw);
    auto expected = expected_func(a, p);

    ASSERT_TRUE(torch::equal(result, expected))
        << fail_message;
}


template <typename T, typename FExpected>
void run_modneg_tc(
    const torch::Tensor& a,
    T p_scalar,
    const std::vector<int64_t>& result_shape,
    void(*kernel)(
        const std::shared_ptr<DeviceTensor<T>>&,
        T,
        std::shared_ptr<DeviceTensor<T>>&),
    FExpected expected_func,
    const std::string& fail_message)
{
    auto a_hw      = host_to_device<T>(a);
    auto result_hw = zeros<T>(result_shape);

    kernel(a_hw, p_scalar, result_hw);
    auto result   = device_to_host<T>(result_hw);
    auto expected = expected_func(a, p_scalar);

    ASSERT_TRUE(torch::equal(result, expected))
        << fail_message;
}

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MOD_MUL_TTT TESTS                                   ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModMulTttTests, Basic1DInt32) {
    auto a_cpu = torch::tensor({1, 2, 3}, torch::kInt32);
    auto b_cpu = torch::tensor({4, 5, 6}, torch::kInt32);
    auto p_cpu = torch::tensor({3, 4, 5}, torch::kInt32);

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {3},
        modmul_ttt<int32_t>,
        expected_ops::modmul,
        "Basic1DInt32 failed"
    );
}

TEST(ModMulTttTests, Basic2DBroadcast) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32);
    auto b_cpu = torch::tensor({10, 20, 30},    torch::kInt32);
    auto p_cpu = torch::tensor({ 7, 8, 9},     torch::kInt32);

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {2,3},
        modmul_ttt<int32_t>,
        expected_ops::modmul,
        "Basic2DBroadcast failed"
    );
}

TEST(ModMulTttTests, Basic1DInt64) {
    std::vector<int64_t> a_data = {10000000000LL, 20000000000LL};
    std::vector<int64_t> b_data = {3LL, 5LL};
    std::vector<int64_t> p_data = {7LL, 9LL};
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto a_cpu = torch::tensor(a_data, opts);
    auto b_cpu = torch::tensor(b_data, opts);
    auto p_cpu = torch::tensor(p_data, opts);

    run_modop_ttt<int64_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {2},
        modmul_ttt<int64_t>,
        expected_ops::modmul,
        "Basic1DInt64 failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

 TEST(ModMulTttTests, NegativeExampleMixedSigns) {
    auto opts  = torch::TensorOptions().dtype(torch::kInt32);
    // a and b both include negatives
    auto a_cpu = torch::tensor({ -1,  -2,   3 }, opts);
    auto b_cpu = torch::tensor({  4,  -5,  -6 }, opts);
    // per‐element moduli
    auto p_cpu = torch::tensor({  7,   8,   9 }, opts);

    // Expected host-side:
    //   (-1 *  4) % 7 = -4 % 7 =  3
    //   (-2 * -5) % 8 = 10 % 8 =  2
    //   ( 3 * -6) % 9 = -18 % 9 =  0
    //
    // So we expect [3, 2, 0].

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {3},
        &modmul_ttt<int32_t>,
        expected_ops::modmul,
        "NegativeExampleMixedSigns failed"
    );
}


TEST(ModMulTttTests, HighRankBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::arange(2*3*4, opts).reshape({2,3,4});
    auto b_cpu = torch::tensor({1,2,3}, opts).reshape({3,1});
    auto p_cpu = torch::tensor({5,6,7,8}, opts);

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {2,3,4},
        modmul_ttt<int32_t>,
        expected_ops::modmul,
        "HighRankBroadcast failed"
    );
}

TEST(ModMulTttTests, ThreeDMixedBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({
        {{1,2,3,4}},
        {{5,6,7,8}}
    }, opts);
    auto b_cpu = torch::tensor({
        {{2},{3},{4}}
    }, opts);
    auto p_cpu = torch::tensor({10,11,12,13}, opts);

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {2,3,4},
        &modmul_ttt<int32_t>,
        expected_ops::modmul,
        "ThreeDMixedBroadcast failed"
    );
}

TEST(ModMulTttTests, Stress3DBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    const int D1 = 10, D2 = 10, D3 = 10;
    auto a_cpu = torch::arange(D1*D2*D3, opts).reshape({D1,D2,D3});
    auto b_cpu = torch::arange(1, D1+1, opts).reshape({D1,1,1});
    auto p_cpu = torch::arange(2, D2+2, opts);

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {D1,D2,D3},
        modmul_ttt<int32_t>,
        expected_ops::modmul,
        "Stress3DBroadcast failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModMulTttTests, Throws_OnAShapeMismatch) {
    auto a_cpu = torch::tensor({1,2},   torch::kInt32);
    auto b_cpu = torch::tensor({3,4,5}, torch::kInt32);
    auto p_cpu = torch::tensor({6,7,8}, torch::kInt32);

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {3},
            modmul_ttt<int32_t>,
            expected_ops::modmul,
            "Should throw on A shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTttTests, Throws_OnBShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5},   torch::kInt32);
    auto p_cpu = torch::tensor({6,7,8}, torch::kInt32);

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {3},
            modmul_ttt<int32_t>,
            expected_ops::modmul,
            "Should throw on B shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTttTests, Throws_OnPShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5,6}, torch::kInt32);
    auto p_cpu = torch::tensor({7,8},   torch::kInt32);

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {3},
            modmul_ttt<int32_t>,
            expected_ops::modmul,
            "Should throw on P shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTttTests, Throws_OnPDimention) {
    auto a_cpu = torch::tensor({{1,2,3},{1,2,3}}, torch::kInt32); // [2×3]
    auto b_cpu = torch::tensor({{4,5,6},{4,5,6}}, torch::kInt32); // [2×3]
    auto p_cpu = torch::tensor({{7,8,9},{7,8,9}}, torch::kInt32); // [2×3]

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {2, 3},
            modmul_ttt<int32_t>,
            expected_ops::modmul,
            "Should throw on P has more than 1 dimension"
    ),
    std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTttTests, Throws_OnResultShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3},   torch::kInt32);
    auto b_cpu = torch::tensor({4,5,6},   torch::kInt32);
    auto p_cpu = torch::tensor({7,8,9},   torch::kInt32);

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {2},  // wrong
            &modmul_ttt<int32_t>,
            expected_ops::modmul,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MOD_MUL_TTC TESTS                                   ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModMulTtcTests, Basic1DInt32) {
    auto a_cpu = torch::tensor({1, 2, 3}, torch::kInt32);
    auto b_cpu = torch::tensor({4, 5, 6}, torch::kInt32);
    int32_t p_scalar = 7;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {3},
        modmul_ttc<int32_t>,
        expected_ops::modmul,
        "Basic1DInt32 (scalar p) failed"
    );
}

TEST(ModMulTtcTests, Basic2DBroadcast) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32); // [2×3]
    auto b_cpu = torch::tensor({10, 20, 30},       torch::kInt32); // [3] → broadcast to [2×3]
    int32_t p_scalar = 11;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {2,3},
        modmul_ttc<int32_t>,
        expected_ops::modmul,
        "Basic2DBroadcast (scalar p) failed"
    );
}

TEST(ModMulTtcTests, Basic1DInt64) {
    std::vector<int64_t> a_data = {10000000000LL, 20000000000LL};
    std::vector<int64_t> b_data = {3LL, 5LL};
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto a_cpu = torch::tensor(a_data, opts);
    auto b_cpu = torch::tensor(b_data, opts);
    int64_t p_scalar = 9LL;

    run_modop_tensor_scalar<int64_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {2},
        modmul_ttc<int64_t>,
        expected_ops::modmul,
        "Basic1DInt64 (scalar p) failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ModMulTtcTests, NegativeExampleMixedSigns) {
    auto opts  = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({ -1, -2,  3 }, opts);
    auto b_cpu = torch::tensor({  4, -5, -6 }, opts);
    int32_t p_scalar = 9;  // one scalar modulus for all elements

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {3},
        &modmul_ttc<int32_t>,
        expected_ops::modmul,
        "NegativeExampleMixedSigns (scalar p) failed"
    );
}

TEST(ModMulTtcTests, HighRankBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::arange(2*3*4, opts).reshape({2,3,4});
    auto b_cpu = torch::tensor({1,2,3}, opts).reshape({3,1}); // broadcast over last two dims
    int32_t p_scalar = 17;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {2,3,4},
        modmul_ttc<int32_t>,
        expected_ops::modmul,
        "HighRankBroadcast (scalar p) failed"
    );
}

TEST(ModMulTtcTests, ThreeDMixedBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({
        {{1,2,3,4}},
        {{5,6,7,8}}
    }, opts);                       // shape [2×1×4]
    auto b_cpu = torch::tensor({
        {{2},{3},{4}}
    }, opts);                       // shape [1×3×1]
    int32_t p_scalar = 19;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {2,3,4},
        &modmul_ttc<int32_t>,
        expected_ops::modmul,
        "ThreeDMixedBroadcast (scalar p) failed"
    );
}

TEST(ModMulTtcTests, Stress3DBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    constexpr int D1 = 10, D2 = 10, D3 = 10;
    auto a_cpu = torch::arange(D1*D2*D3, opts).reshape({D1,D2,D3});
    auto b_cpu = torch::arange(1, D1+1, opts).reshape({D1,1,1});   // broadcast across D2×D3
    int32_t p_scalar = 9973;                                        // a large prime

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {D1,D2,D3},
        modmul_ttc<int32_t>,
        expected_ops::modmul,
        "Stress3DBroadcast (scalar p) failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModMulTtcTests, Throws_OnAShapeMismatch) {
    auto a_cpu = torch::tensor({1,2},   torch::kInt32);
    auto b_cpu = torch::tensor({3,4,5}, torch::kInt32);
    int32_t p_scalar = 7;

    EXPECT_THROW(
        run_modop_tensor_scalar<int32_t>(
            a_cpu, b_cpu, p_scalar,
            /* result_shape */ {3},
            modmul_ttc<int32_t>,
            expected_ops::modmul,
            "Should throw on A shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTtcTests, Throws_OnBShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5},   torch::kInt32);
    int32_t p_scalar = 11;

    EXPECT_THROW(
        run_modop_tensor_scalar<int32_t>(
            a_cpu, b_cpu, p_scalar,
            /* result_shape */ {3},
            modmul_ttc<int32_t>,
            expected_ops::modmul,
            "Should throw on B shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTtcTests, Throws_OnResultShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5,6}, torch::kInt32);
    int32_t p_scalar = 13;

    EXPECT_THROW(
        run_modop_tensor_scalar<int32_t>(
            a_cpu, b_cpu, p_scalar,
            /* result_shape */ {2},  // wrong
            &modmul_ttc<int32_t>,
            expected_ops::modmul,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MOD_MUL_TCT TESTS                                   ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModMulTctTests, Basic1DInt32) {
    auto a_cpu = torch::tensor({1, 2, 3}, torch::kInt32);
    int32_t b_scalar = 4;
    auto  p_cpu = torch::tensor({3, 4, 5}, torch::kInt32);

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {3},
        modmul_tct<int32_t>,
        expected_ops::modmul,
        "Basic1DInt32 (scalar b) failed"
    );
}

TEST(ModMulTctTests, Basic2DBroadcast) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32);  // [2×3]
    int32_t b_scalar = 10;
    auto  p_cpu = torch::tensor({7, 8, 9}, torch::kInt32);         // [3] → broadcast

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {2,3},
        modmul_tct<int32_t>,
        expected_ops::modmul,
        "Basic2DBroadcast (scalar b) failed"
    );
}

TEST(ModMulTctTests, Basic1DInt64) {
    std::vector<int64_t> a_data = {10000000000LL, 20000000000LL};
    int64_t b_scalar = 3LL;
    std::vector<int64_t> p_data = {7LL, 9LL};
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto a_cpu = torch::tensor(a_data, opts);
    auto  p_cpu = torch::tensor(p_data, opts);

    run_modop_scalar_tensor<int64_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {2},
        modmul_tct<int64_t>,
        expected_ops::modmul,
        "Basic1DInt64 (scalar b) failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ModMulTctTests, NegativeExampleMixedSigns) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({ -1, -2,  3 }, opts);
    int32_t b_scalar = -5;
    auto  p_cpu = torch::tensor({ 7, 8, 9 }, opts);

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {3},
        &modmul_tct<int32_t>,
        expected_ops::modmul,
        "NegativeExampleMixedSigns (scalar b) failed"
    );
}

TEST(ModMulTctTests, HighRankBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::arange(2*3*4, opts).reshape({2,3,4});
    int32_t b_scalar = 3;
    auto  p_cpu = torch::tensor({5,6,7,8}, opts);   // length 4 → broadcast to last dim

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {2,3,4},
        modmul_tct<int32_t>,
        expected_ops::modmul,
        "HighRankBroadcast (scalar b) failed"
    );
}

TEST(ModMulTctTests, ThreeDMixedBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({
        {{1,2,3,4}},
        {{5,6,7,8}}
    }, opts);                                    // [2×1×4]
    int32_t b_scalar = 3;
    auto  p_cpu = torch::tensor({10,11,12,13}, opts);

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {2,1,4},
        &modmul_tct<int32_t>,
        expected_ops::modmul,
        "ThreeDMixedBroadcast (scalar b) failed"
    );
}

TEST(ModMulTctTests, Stress3DBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    constexpr int D1 = 10, D2 = 10, D3 = 10;
    auto a_cpu = torch::arange(D1*D2*D3, opts).reshape({D1,D2,D3});
    int32_t b_scalar = 17;
    auto  p_cpu = torch::arange(2, D2+2, opts);       // length 10 → broadcast to middle dim

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {D1,D2,D3},
        modmul_tct<int32_t>,
        expected_ops::modmul,
        "Stress3DBroadcast (scalar b) failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModMulTctTests, Throws_OnAShapeMismatch) {
    auto a_cpu = torch::tensor({1,2},   torch::kInt32);  // len 2
    int32_t b_scalar = 7;
    auto  p_cpu = torch::tensor({3,4,5}, torch::kInt32); // len 3

    EXPECT_THROW(
        run_modop_scalar_tensor<int32_t>(
            a_cpu, b_scalar, p_cpu,
            /* result_shape */ {3},
            modmul_tct<int32_t>,
            expected_ops::modmul,
            "Should throw on A shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTctTests, Throws_OnPShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32); // len 3
    int32_t b_scalar = 4;
    auto  p_cpu = torch::tensor({5,6},   torch::kInt32); // len 2 → mismatch

    EXPECT_THROW(
        run_modop_scalar_tensor<int32_t>(
            a_cpu, b_scalar, p_cpu,
            /* result_shape */ {3},
            modmul_tct<int32_t>,
            expected_ops::modmul,
            "Should throw on P shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTctTests, Throws_OnPDimention) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32); // [2×3]
    int32_t b_scalar = 4;
    auto  p_cpu = torch::tensor({{7,8,9},{7,8,9}}, torch::kInt32); // 2-D → invalid

    EXPECT_THROW(
        run_modop_scalar_tensor<int32_t>(
            a_cpu, b_scalar, p_cpu,
            /* result_shape */ {2,3},
            modmul_tct<int32_t>,
            expected_ops::modmul,
            "Should throw when P has more than 1 dimension"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTctTests, Throws_OnResultShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    int32_t b_scalar = 5;
    auto  p_cpu = torch::tensor({6,7,8}, torch::kInt32);

    EXPECT_THROW(
        run_modop_scalar_tensor<int32_t>(
            a_cpu, b_scalar, p_cpu,
            /* result_shape */ {2},  // wrong
            &modmul_tct<int32_t>,
            expected_ops::modmul,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MOD_MUL_TCC TESTS                                   ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModMulTccTests, Basic1DInt32) {
    auto a_cpu = torch::tensor({1, 2, 3}, torch::kInt32);
    int32_t b_scalar = 4;
    int32_t p_scalar = 7;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {3},
        modmul_tcc<int32_t>,
        expected_ops::modmul,
        "Basic1DInt32 (scalar b & p) failed"
    );
}

TEST(ModMulTccTests, Basic2D) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32); // [2×3]
    int32_t b_scalar = 10;
    int32_t p_scalar = 11;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {2,3},
        modmul_tcc<int32_t>,
        expected_ops::modmul,
        "Basic2D (scalar b & p) failed"
    );
}

TEST(ModMulTccTests, Basic1DInt64) {
    std::vector<int64_t> a_data = {10000000000LL, 20000000000LL};
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto a_cpu = torch::tensor(a_data, opts);
    int64_t b_scalar = 3LL;
    int64_t p_scalar = 9LL;

    run_modop_scalar_scalar<int64_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {2},
        modmul_tcc<int64_t>,
        expected_ops::modmul,
        "Basic1DInt64 (scalar b & p) failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ModMulTccTests, NegativeExampleMixedSigns) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({ -1, -2,  3 }, opts);
    int32_t b_scalar = -5;
    int32_t p_scalar = 9;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {3},
        &modmul_tcc<int32_t>,
        expected_ops::modmul,
        "NegativeExampleMixedSigns (scalar b & p) failed"
    );
}

TEST(ModMulTccTests, HighRank) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::arange(2*3*4, opts).reshape({2,3,4});
    int32_t b_scalar = 3;
    int32_t p_scalar = 17;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {2,3,4},
        modmul_tcc<int32_t>,
        expected_ops::modmul,
        "HighRank (scalar b & p) failed"
    );
}

TEST(ModMulTccTests, ThreeDMixed) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({
        {{1,2,3,4}},
        {{5,6,7,8}}
    }, opts);                       // [2×1×4]
    int32_t b_scalar = 3;
    int32_t p_scalar = 19;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {2,1,4},
        &modmul_tcc<int32_t>,
        expected_ops::modmul,
        "ThreeDMixed (scalar b & p) failed"
    );
}

TEST(ModMulTccTests, Stress3D) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    constexpr int D1 = 10, D2 = 10, D3 = 10;
    auto a_cpu = torch::arange(D1*D2*D3, opts).reshape({D1,D2,D3});
    int32_t b_scalar = 17;
    int32_t p_scalar = 9973;   // large prime

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {D1,D2,D3},
        modmul_tcc<int32_t>,
        expected_ops::modmul,
        "Stress3D (scalar b & p) failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModMulTccTests, Throws_OnAShapeMismatch) {
    auto a_cpu = torch::tensor({1,2}, torch::kInt32);   // len 2
    int32_t b_scalar = 7;
    int32_t p_scalar = 13;

    EXPECT_THROW(
        run_modop_scalar_scalar<int32_t>(
            a_cpu, b_scalar, p_scalar,
            /* result_shape */ {3},  // expected len 3
            modmul_tcc<int32_t>,
            expected_ops::modmul,
            "Should throw on A shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModMulTccTests, Throws_OnResultShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32); // len 3
    int32_t b_scalar = 5;
    int32_t p_scalar = 6;

    EXPECT_THROW(
        run_modop_scalar_scalar<int32_t>(
            a_cpu, b_scalar, p_scalar,
            /* result_shape */ {2},  // wrong
            &modmul_tcc<int32_t>,
            expected_ops::modmul,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}



/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MOD_SUM_TTT TESTS                                   ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModSumTttTests, Basic1DInt32) {
    auto a_cpu = torch::tensor({1, 2, 3}, torch::kInt32);
    auto b_cpu = torch::tensor({4, 5, 6}, torch::kInt32);
    auto p_cpu = torch::tensor({3, 4, 5}, torch::kInt32);

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {3},
        modsum_ttt<int32_t>,
        expected_ops::modsum,
        "Basic1DInt32 failed"
    );
}

TEST(ModSumTttTests, Basic2DBroadcast) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32);
    auto b_cpu = torch::tensor({10, 20, 30},       torch::kInt32);   // [3] → broadcast
    auto p_cpu = torch::tensor({ 7,  8,  9},       torch::kInt32);

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {2,3},
        modsum_ttt<int32_t>,
        expected_ops::modsum,
        "Basic2DBroadcast failed"
    );
}

TEST(ModSumTttTests, Basic1DInt64) {
    std::vector<int64_t> a_data = {10000000000LL, 20000000000LL};
    std::vector<int64_t> b_data = {3LL, 5LL};
    std::vector<int64_t> p_data = {7LL, 9LL};
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto a_cpu = torch::tensor(a_data, opts);
    auto b_cpu = torch::tensor(b_data, opts);
    auto p_cpu = torch::tensor(p_data, opts);

    run_modop_ttt<int64_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {2},
        modsum_ttt<int64_t>,
        expected_ops::modsum,
        "Basic1DInt64 failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ModSumTttTests, NegativeExampleMixedSigns) {
    auto opts  = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({ -1,  -2,   3 }, opts);
    auto b_cpu = torch::tensor({  4,  -5,  -6 }, opts);
    auto p_cpu = torch::tensor({  7,   8,   9 }, opts);

    // Expected host-side:
    //   (-1 +  4) % 7 =  3
    //   (-2 + -5) % 8 =  1
    //   ( 3 + -6) % 9 =  6  → [3,1,6]

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {3},
        &modsum_ttt<int32_t>,
        expected_ops::modsum,
        "NegativeExampleMixedSigns failed"
    );
}

TEST(ModSumTttTests, HighRankBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::arange(2*3*4, opts).reshape({2,3,4});
    auto b_cpu = torch::tensor({1,2,3}, opts).reshape({3,1});   // broadcast over last two dims
    auto p_cpu = torch::tensor({5,6,7,8}, opts);                 // mod per-column

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {2,3,4},
        modsum_ttt<int32_t>,
        expected_ops::modsum,
        "HighRankBroadcast failed"
    );
}

TEST(ModSumTttTests, ThreeDMixedBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({
        {{1,2,3,4}},
        {{5,6,7,8}}
    }, opts);                                // [2×1×4]
    auto b_cpu = torch::tensor({
        {{2},{3},{4}}
    }, opts);                                // [1×3×1]
    auto p_cpu = torch::tensor({10,11,12,13}, opts);

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {2,3,4},
        &modsum_ttt<int32_t>,
        expected_ops::modsum,
        "ThreeDMixedBroadcast failed"
    );
}

TEST(ModSumTttTests, Stress3DBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    const int D1 = 10, D2 = 10, D3 = 10;
    auto a_cpu = torch::arange(D1*D2*D3, opts).reshape({D1,D2,D3});
    auto b_cpu = torch::arange(1, D1+1, opts).reshape({D1,1,1});  // broadcast across D2×D3
    auto p_cpu = torch::arange(2, D2+2, opts);                     // length 10 → broadcast middle dim

    run_modop_ttt<int32_t>(
        a_cpu, b_cpu, p_cpu,
        /* result_shape */ {D1,D2,D3},
        modsum_ttt<int32_t>,
        expected_ops::modsum,
        "Stress3DBroadcast failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModSumTttTests, Throws_OnAShapeMismatch) {
    auto a_cpu = torch::tensor({1,2},   torch::kInt32);
    auto b_cpu = torch::tensor({3,4,5}, torch::kInt32);   // incompatible
    auto p_cpu = torch::tensor({6,7,8}, torch::kInt32);

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {3},
            modsum_ttt<int32_t>,
            expected_ops::modsum,
            "Should throw on A shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTttTests, Throws_OnBShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5},   torch::kInt32);   // incompatible
    auto p_cpu = torch::tensor({6,7,8}, torch::kInt32);

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {3},
            modsum_ttt<int32_t>,
            expected_ops::modsum,
            "Should throw on B shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTttTests, Throws_OnPShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5,6}, torch::kInt32);
    auto p_cpu = torch::tensor({7,8},   torch::kInt32);   // wrong length

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {3},
            modsum_ttt<int32_t>,
            expected_ops::modsum,
            "Should throw on P shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTttTests, Throws_OnPDimention) {
    auto a_cpu = torch::tensor({{1,2,3},{1,2,3}}, torch::kInt32); // [2×3]
    auto b_cpu = torch::tensor({{4,5,6},{4,5,6}}, torch::kInt32); // [2×3]
    auto p_cpu = torch::tensor({{7,8,9},{7,8,9}}, torch::kInt32); // 2-D → invalid

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {2,3},
            modsum_ttt<int32_t>,
            expected_ops::modsum,
            "Should throw on P has more than 1 dimension"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTttTests, Throws_OnResultShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5,6}, torch::kInt32);
    auto p_cpu = torch::tensor({7,8,9}, torch::kInt32);

    EXPECT_THROW(
        run_modop_ttt<int32_t>(
            a_cpu, b_cpu, p_cpu,
            /* result_shape */ {2},  // wrong
            &modsum_ttt<int32_t>,
            expected_ops::modsum,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MOD_SUM_TTC TESTS                                   ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModSumTtcTests, Basic1DInt32) {
    auto a_cpu = torch::tensor({1, 2, 3}, torch::kInt32);
    auto b_cpu = torch::tensor({4, 5, 6}, torch::kInt32);
    int32_t p_scalar = 7;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {3},
        modsum_ttc<int32_t>,
        expected_ops::modsum,
        "Basic1DInt32 (scalar p) failed"
    );
}

TEST(ModSumTtcTests, Basic2DBroadcast) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32);  // [2×3]
    auto b_cpu = torch::tensor({10, 20, 30},       torch::kInt32);  // [3] → broadcast
    int32_t p_scalar = 11;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {2,3},
        modsum_ttc<int32_t>,
        expected_ops::modsum,
        "Basic2DBroadcast (scalar p) failed"
    );
}

TEST(ModSumTtcTests, Basic1DInt64) {
    std::vector<int64_t> a_data = {10000000000LL, 20000000000LL};
    std::vector<int64_t> b_data = {3LL, 5LL};
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto a_cpu = torch::tensor(a_data, opts);
    auto b_cpu = torch::tensor(b_data, opts);
    int64_t p_scalar = 9LL;

    run_modop_tensor_scalar<int64_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {2},
        modsum_ttc<int64_t>,
        expected_ops::modsum,
        "Basic1DInt64 (scalar p) failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ModSumTtcTests, NegativeExampleMixedSigns) {
    auto opts  = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({ -1,  -2,   3 }, opts);
    auto b_cpu = torch::tensor({  4,  -5,  -6 }, opts);
    int32_t p_scalar = 9;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {3},
        &modsum_ttc<int32_t>,
        expected_ops::modsum,
        "NegativeExampleMixedSigns (scalar p) failed"
    );
}

TEST(ModSumTtcTests, HighRankBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::arange(2*3*4, opts).reshape({2,3,4});
    auto b_cpu = torch::tensor({1,2,3}, opts).reshape({3,1});   // broadcast over last two dims
    int32_t p_scalar = 17;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {2,3,4},
        modsum_ttc<int32_t>,
        expected_ops::modsum,
        "HighRankBroadcast (scalar p) failed"
    );
}

TEST(ModSumTtcTests, ThreeDMixedBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({
        {{1,2,3,4}},
        {{5,6,7,8}}
    }, opts);                                    // [2×1×4]
    auto b_cpu = torch::tensor({
        {{2},{3},{4}}
    }, opts);                                    // [1×3×1]
    int32_t p_scalar = 19;

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {2,3,4},
        &modsum_ttc<int32_t>,
        expected_ops::modsum,
        "ThreeDMixedBroadcast (scalar p) failed"
    );
}

TEST(ModSumTtcTests, Stress3DBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    constexpr int D1 = 10, D2 = 10, D3 = 10;
    auto a_cpu = torch::arange(D1*D2*D3, opts).reshape({D1,D2,D3});
    auto b_cpu = torch::arange(1, D1+1, opts).reshape({D1,1,1});  // broadcast across D2×D3
    int32_t p_scalar = 9973;                                      // large prime

    run_modop_tensor_scalar<int32_t>(
        a_cpu, b_cpu, p_scalar,
        /* result_shape */ {D1,D2,D3},
        modsum_ttc<int32_t>,
        expected_ops::modsum,
        "Stress3DBroadcast (scalar p) failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModSumTtcTests, Throws_OnAShapeMismatch) {
    auto a_cpu = torch::tensor({1,2},   torch::kInt32);  // len 2
    auto b_cpu = torch::tensor({3,4,5}, torch::kInt32);  // len 3 → incompatible
    int32_t p_scalar = 7;

    EXPECT_THROW(
        run_modop_tensor_scalar<int32_t>(
            a_cpu, b_cpu, p_scalar,
            /* result_shape */ {3},
            modsum_ttc<int32_t>,
            expected_ops::modsum,
            "Should throw on A shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTtcTests, Throws_OnBShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5},   torch::kInt32);  // len 2 → incompatible
    int32_t p_scalar = 11;

    EXPECT_THROW(
        run_modop_tensor_scalar<int32_t>(
            a_cpu, b_cpu, p_scalar,
            /* result_shape */ {3},
            modsum_ttc<int32_t>,
            expected_ops::modsum,
            "Should throw on B shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTtcTests, Throws_OnResultShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    auto b_cpu = torch::tensor({4,5,6}, torch::kInt32);
    int32_t p_scalar = 13;

    EXPECT_THROW(
        run_modop_tensor_scalar<int32_t>(
            a_cpu, b_cpu, p_scalar,
            /* result_shape */ {2},  // wrong
            &modsum_ttc<int32_t>,
            expected_ops::modsum,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MOD_SUM_TCT TESTS                                   ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModSumTctTests, Basic1DInt32) {
    auto a_cpu = torch::tensor({1, 2, 3}, torch::kInt32);
    int32_t b_scalar = 4;
    auto  p_cpu = torch::tensor({3, 4, 5}, torch::kInt32);

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {3},
        modsum_tct<int32_t>,
        expected_ops::modsum,
        "Basic1DInt32 (scalar b) failed"
    );
}

TEST(ModSumTctTests, Basic2DBroadcast) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32);  // [2×3]
    int32_t b_scalar = 10;
    auto  p_cpu = torch::tensor({7, 8, 9}, torch::kInt32);         // [3] → broadcast

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {2,3},
        modsum_tct<int32_t>,
        expected_ops::modsum,
        "Basic2DBroadcast (scalar b) failed"
    );
}

TEST(ModSumTctTests, Basic1DInt64) {
    std::vector<int64_t> a_data = {10000000000LL, 20000000000LL};
    int64_t b_scalar = 3LL;
    std::vector<int64_t> p_data = {7LL, 9LL};
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto a_cpu = torch::tensor(a_data, opts);
    auto  p_cpu = torch::tensor(p_data, opts);

    run_modop_scalar_tensor<int64_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {2},
        modsum_tct<int64_t>,
        expected_ops::modsum,
        "Basic1DInt64 (scalar b) failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ModSumTctTests, NegativeExampleMixedSigns) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({ -1, -2,  3 }, opts);
    int32_t b_scalar = -5;
    auto  p_cpu = torch::tensor({ 7, 8, 9 }, opts);

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {3},
        &modsum_tct<int32_t>,
        expected_ops::modsum,
        "NegativeExampleMixedSigns (scalar b) failed"
    );
}

TEST(ModSumTctTests, HighRankBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::arange(2*3*4, opts).reshape({2,3,4});
    int32_t b_scalar = 3;
    auto  p_cpu = torch::tensor({5,6,7,8}, opts);  // length 4 → broadcast to last dim

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {2,3,4},
        modsum_tct<int32_t>,
        expected_ops::modsum,
        "HighRankBroadcast (scalar b) failed"
    );
}

TEST(ModSumTctTests, ThreeDMixedBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({
        {{1,2,3,4}},
        {{5,6,7,8}}
    }, opts);                              // [2×1×4]
    int32_t b_scalar = 3;
    auto  p_cpu = torch::tensor({10,11,12,13}, opts);

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {2,1,4},
        &modsum_tct<int32_t>,
        expected_ops::modsum,
        "ThreeDMixedBroadcast (scalar b) failed"
    );
}

TEST(ModSumTctTests, Stress3DBroadcast) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    constexpr int D1 = 10, D2 = 10, D3 = 10;
    auto a_cpu = torch::arange(D1*D2*D3, opts).reshape({D1,D2,D3});
    int32_t b_scalar = 17;
    auto  p_cpu = torch::arange(2, D2+2, opts);          // length 10 → broadcast middle dim

    run_modop_scalar_tensor<int32_t>(
        a_cpu, b_scalar, p_cpu,
        /* result_shape */ {D1,D2,D3},
        modsum_tct<int32_t>,
        expected_ops::modsum,
        "Stress3DBroadcast (scalar b) failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModSumTctTests, Throws_OnAShapeMismatch) {
    auto a_cpu = torch::tensor({1,2},   torch::kInt32);   // len 2
    int32_t b_scalar = 7;
    auto  p_cpu = torch::tensor({3,4,5}, torch::kInt32);  // len 3

    EXPECT_THROW(
        run_modop_scalar_tensor<int32_t>(
            a_cpu, b_scalar, p_cpu,
            /* result_shape */ {3},
            modsum_tct<int32_t>,
            expected_ops::modsum,
            "Should throw on A shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTctTests, Throws_OnPShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32); // len 3
    int32_t b_scalar = 4;
    auto  p_cpu = torch::tensor({5,6},   torch::kInt32); // len 2 → mismatch

    EXPECT_THROW(
        run_modop_scalar_tensor<int32_t>(
            a_cpu, b_scalar, p_cpu,
            /* result_shape */ {3},
            modsum_tct<int32_t>,
            expected_ops::modsum,
            "Should throw on P shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTctTests, Throws_OnPDimention) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32); // [2×3]
    int32_t b_scalar = 4;
    auto  p_cpu = torch::tensor({{7,8,9},{7,8,9}}, torch::kInt32); // 2-D → invalid

    EXPECT_THROW(
        run_modop_scalar_tensor<int32_t>(
            a_cpu, b_scalar, p_cpu,
            /* result_shape */ {2,3},
            modsum_tct<int32_t>,
            expected_ops::modsum,
            "Should throw when P has more than 1 dimension"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTctTests, Throws_OnResultShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32);
    int32_t b_scalar = 5;
    auto  p_cpu = torch::tensor({6,7,8}, torch::kInt32);

    EXPECT_THROW(
        run_modop_scalar_tensor<int32_t>(
            a_cpu, b_scalar, p_cpu,
            /* result_shape */ {2},  // wrong
            &modsum_tct<int32_t>,
            expected_ops::modsum,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MOD_SUM_TCC TESTS                                   ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModSumTccTests, Basic1DInt32) {
    auto a_cpu = torch::tensor({1, 2, 3}, torch::kInt32);
    int32_t b_scalar = 4;
    int32_t p_scalar = 7;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {3},
        modsum_tcc<int32_t>,
        expected_ops::modsum,
        "Basic1DInt32 (scalar b & p) failed"
    );
}

TEST(ModSumTccTests, Basic2D) {
    auto a_cpu = torch::tensor({{1,2,3},{4,5,6}}, torch::kInt32); // [2×3]
    int32_t b_scalar = 10;
    int32_t p_scalar = 11;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {2,3},
        modsum_tcc<int32_t>,
        expected_ops::modsum,
        "Basic2D (scalar b & p) failed"
    );
}

TEST(ModSumTccTests, Basic1DInt64) {
    std::vector<int64_t> a_data = {10000000000LL, 20000000000LL};
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto a_cpu = torch::tensor(a_data, opts);
    int64_t b_scalar = 3LL;
    int64_t p_scalar = 9LL;

    run_modop_scalar_scalar<int64_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {2},
        modsum_tcc<int64_t>,
        expected_ops::modsum,
        "Basic1DInt64 (scalar b & p) failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ModSumTccTests, NegativeExampleMixedSigns) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({ -1, -2,  3 }, opts);
    int32_t b_scalar = -5;
    int32_t p_scalar = 9;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {3},
        &modsum_tcc<int32_t>,
        expected_ops::modsum,
        "NegativeExampleMixedSigns (scalar b & p) failed"
    );
}

TEST(ModSumTccTests, HighRank) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::arange(2*3*4, opts).reshape({2,3,4});
    int32_t b_scalar = 3;
    int32_t p_scalar = 17;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {2,3,4},
        modsum_tcc<int32_t>,
        expected_ops::modsum,
        "HighRank (scalar b & p) failed"
    );
}

TEST(ModSumTccTests, ThreeDMixed) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({
        {{1,2,3,4}},
        {{5,6,7,8}}
    }, opts);                       // [2×1×4]
    int32_t b_scalar = 3;
    int32_t p_scalar = 19;

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {2,1,4},
        &modsum_tcc<int32_t>,
        expected_ops::modsum,
        "ThreeDMixed (scalar b & p) failed"
    );
}

TEST(ModSumTccTests, Stress3D) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    constexpr int D1 = 10, D2 = 10, D3 = 10;
    auto a_cpu = torch::arange(D1*D2*D3, opts).reshape({D1,D2,D3});
    int32_t b_scalar = 17;
    int32_t p_scalar = 9973;   // large prime

    run_modop_scalar_scalar<int32_t>(
        a_cpu, b_scalar, p_scalar,
        /* result_shape */ {D1,D2,D3},
        modsum_tcc<int32_t>,
        expected_ops::modsum,
        "Stress3D (scalar b & p) failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModSumTccTests, Throws_OnAShapeMismatch) {
    auto a_cpu = torch::tensor({1,2}, torch::kInt32);   // len 2
    int32_t b_scalar = 7;
    int32_t p_scalar = 13;

    EXPECT_THROW(
        run_modop_scalar_scalar<int32_t>(
            a_cpu, b_scalar, p_scalar,
            /* result_shape */ {3},  // expected len 3
            modsum_tcc<int32_t>,
            expected_ops::modsum,
            "Should throw on A shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModSumTccTests, Throws_OnResultShapeMismatch) {
    auto a_cpu = torch::tensor({1,2,3}, torch::kInt32); // len 3
    int32_t b_scalar = 5;
    int32_t p_scalar = 6;

    EXPECT_THROW(
        run_modop_scalar_scalar<int32_t>(
            a_cpu, b_scalar, p_scalar,
            /* result_shape */ {2},  // wrong
            &modsum_tcc<int32_t>,
            expected_ops::modsum,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}



/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */
/******************************************************************************************************************************* */

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MODNEG_TT TESTS                                     ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ModNegTtTests, Basic1DInt32) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({1, 2, 3}, opts);
    auto p_cpu = torch::tensor({5, 4, 6}, opts);

    // expected: [(-1)%5=4, (-2)%4=2, (-3)%6=3]
    run_modneg_tt<int32_t>(
        a_cpu, p_cpu,
        /* result_shape */ {3},
        &modneg_tt<int32_t>,
        expected_ops::modneg,
        "Basic1DInt32 failed"
    );
}

TEST(ModNegTtTests, Basic2DInt32) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({{ 1, -1 }, { 3, -3 }}, opts);
    auto p_cpu = torch::tensor({{ 7,  8 }, {10,  9 }}, opts);

    // expected: [[6, 7], [7, 6]] because [-1%7=6, 1%8=1→8-1=7 etc]
    run_modneg_tt<int32_t>(
        a_cpu, p_cpu,
        /* result_shape */ {2,2},
        &modneg_tt<int32_t>,
        expected_ops::modneg,
        "Basic2DInt32 failed"
    );
}

TEST(ModNegTtTests, Basic1DInt64) {
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<int64_t> a_data = { -10LL,  20LL, -30LL };
    std::vector<int64_t> p_data = {  7LL,   9LL,  11LL };
    auto a_cpu = torch::tensor(a_data, opts);
    auto p_cpu = torch::tensor(p_data, opts);

    // expected: [(-(-10)%7)=3, (-20)%9= -2→7, (-(-30)%11)=8]
    run_modneg_tt<int64_t>(
        a_cpu, p_cpu,
        /* result_shape */ {3},
        &modneg_tt<int64_t>,
        expected_ops::modneg,
        "Basic1DInt64 failed"
    );
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ModNegTtTests, NegativeExampleMixedSigns) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({ -1,   0,  5, -7 }, opts);
    auto p_cpu = torch::tensor({  7,   3, 10,  8 }, opts);

    // expected: [(-(-1)%7)=1→6? Actually -1%7 = -1 -> +7 = 6
    //            -(0)%3=0,
    //            -(5)%10 = -5->+10=5,
    //            -(-7)%8 = 7 %8 =7]
    run_modneg_tt<int32_t>(
        a_cpu, p_cpu,
        /* result_shape */ {4},
        &modneg_tt<int32_t>,
        expected_ops::modneg,
        "NegativeExampleMixedSigns failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModNegTtTests, Throws_OnShapeMismatch) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({1,2}, opts);
    auto p_cpu = torch::tensor({3,4,5}, opts);

    EXPECT_THROW(
        run_modneg_tt<int32_t>(
            a_cpu, p_cpu,
            /* result_shape */ {3},
            &modneg_tt<int32_t>,
            expected_ops::modneg,
            "Should throw on a/p shape mismatch"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModNegTtTests, Throws_OnPNonPositive) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    // contains zero and negative
    auto a_cpu = torch::tensor({1,2,3}, opts);
    auto p_cpu = torch::tensor({5,0,-7}, opts);

    EXPECT_THROW(
        run_modneg_tt<int32_t>(
            a_cpu, p_cpu,
            /* result_shape */ {3},
            &modneg_tt<int32_t>,
            expected_ops::modneg,
            "Should throw when p has non-positive entries"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModNegTtTests, Throws_OnResultShapeMismatch) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({1,2,3}, opts);
    auto p_cpu = torch::tensor({4,5,6}, opts);

    EXPECT_THROW(
        run_modneg_tt<int32_t>(
            a_cpu, p_cpu,
            /* result_shape */ {2},  // wrong
            &modneg_tt<int32_t>,
            expected_ops::modneg,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            MODNEG_TC TESTS                                     ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

TEST(ModNegTcTests, Basic1DInt32) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({1, 2, -3, 0}, opts);
    int32_t p_scalar = 5;

    // expected: [(-1)%5=4, (-2)%5=3, -(-3)%5=3, -(0)%5=0]
    run_modneg_tc<int32_t>(
        a_cpu, p_scalar,
        /* result_shape */ {4},
        &modneg_tc<int32_t>,
        expected_ops::modneg,
        "Basic1DInt32_scalar failed"
    );
}

TEST(ModNegTcTests, Basic2DInt64) {
    auto opts = torch::TensorOptions().dtype(torch::kInt64);

    // flatten your data into a single vector
    std::vector<int64_t> a_flat = {
        10LL, -10LL,
         0LL,   1LL
    };
    // build a 1-D tensor and then reshape to 2×2
    auto a_cpu = torch::tensor(a_flat, opts).reshape({2, 2});

    int64_t p_scalar = 7LL;

    run_modneg_tc<int64_t>(
        a_cpu, p_scalar,
        /* result_shape */ {2,2},
        &modneg_tc<int64_t>,
        expected_ops::modneg,
        "Basic2DInt64_scalar failed"
    );
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(ModNegTcTests, Throws_OnPScalarNonPositive) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({1,2,3}, opts);
    int32_t p_scalar = 0;

    EXPECT_THROW(
        run_modneg_tc<int32_t>(
            a_cpu, p_scalar,
            /* result_shape */ {3},
            &modneg_tc<int32_t>,
            expected_ops::modneg,
            "Should throw on non-positive scalar modulus"
        ),
        std::invalid_argument
    );
}

VALIDATION_TEST(ModNegTcTests, Throws_OnResultShapeMismatch) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto a_cpu = torch::tensor({1,2,3}, opts);
    int32_t p_scalar = 5;

    EXPECT_THROW(
        run_modneg_tc<int32_t>(
            a_cpu, p_scalar,
            /* result_shape */ {2},  // wrong
            &modneg_tc<int32_t>,
            expected_ops::modneg,
            "Should throw on result shape mismatch"
        ),
        std::invalid_argument
    );
}
