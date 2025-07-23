#include "tensor_advanced_ops.h"
#include "memory_management.h"
#include "gtest/gtest.h"

using namespace lattica_hw_api;

TEST(GDecompositionEdgeCases, ScalarValues) {
    torch::Tensor a_cpu = torch::tensor({0, 1, 2, 3}, torch::dtype(torch::kInt32));
    int64_t power = 2;
    int64_t base_bits = 1;  // base = 2

    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = empty<int32_t>({4, power});
    apply_g_decomp<int32_t>(a_hw, result_hw, power, base_bits);

    torch::Tensor expected = torch::tensor({
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 1}
    }, torch::dtype(torch::kInt32));

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(GDecompositionEdgeCases, ZeroInput) {
    torch::Tensor a_cpu = torch::zeros({5}, torch::dtype(torch::kInt32));
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = empty<int32_t>({5, 4});
    apply_g_decomp<int32_t>(a_hw, result_hw, 4, 2);  // base = 4

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::all(result == 0).item<bool>());
}

TEST(GDecompositionEdgeCases, MultiDimensionalInput) {
    torch::Tensor a_cpu = torch::tensor({
        {{5, 12}, {3, 1}},
        {{8, 7}, {9, 2}}
    }, torch::dtype(torch::kInt32));  // [2, 2, 2]
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = empty<int32_t>({2, 2, 2, 3});  // [2,2,2,3]

    apply_g_decomp<int32_t>(a_hw, result_hw, 3, 2);  // base = 4

    torch::Tensor expected = torch::tensor({
        {
            { {1, 1, 0}, {0, 3, 0} },
            { {3, 0, 0}, {1, 0, 0} }
        },
        {
            { {0, 2, 0}, {3, 1, 0} },
            { {1, 2, 0}, {2, 0, 0} }
        }
    }, torch::dtype(torch::kInt32));

    auto result = device_to_host<int32_t>(result_hw);
    ASSERT_TRUE(torch::equal(result, expected));
}

TEST(GDecompositionEdgeCases, OverflowWarningCheck) {
    torch::Tensor a_cpu = torch::tensor({255}, torch::dtype(torch::kInt32));
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = empty<int32_t>({1, 3});

    testing::internal::CaptureStderr();
    apply_g_decomp<int32_t>(a_hw, result_hw, 3, 3);  // base = 8, max representable = 512
    std::string output = testing::internal::GetCapturedStderr();

    ASSERT_EQ(output.find("exceeds representation capacity"), std::string::npos);
}

TEST(GDecompositionEdgeCases, InvalidShapeMismatch) {
    torch::Tensor a_cpu = torch::tensor({10, 20}, torch::dtype(torch::kInt32));
    auto a_hw = host_to_device<int32_t>(a_cpu);
    auto result_hw = empty<int32_t>({3, 2});

    EXPECT_THROW(
        apply_g_decomp<int32_t>(a_hw, result_hw, 2, 2),
        std::invalid_argument
    );
}

/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                        TAKE_ALONG_AXIS TESTS                                   ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

// 1D gather along axis 0
TEST(TakeAlongAxisTests, Basic1D) {
    auto t = torch::tensor({5, 10, 15, 20}, torch::kInt64);
    auto idx = torch::tensor({2, 0, 3, 1}, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 0);

    auto hw_t = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    // allocate output of shape {idx.size(0)}
    auto hw_out = empty<int64_t>({ idx.size(0) });

    take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out);
    auto out = device_to_host<int64_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

TEST(TakeAlongAxisTests, Basic1D_Int32) {
  auto t = torch::tensor({5, 10, 15, 20}, torch::kInt32);
  auto idx = torch::tensor({2, 0, 3, 1}, torch::kInt64);
  // torch::take_along_dim expects indices to be int64, so cast idx if needed
  auto idx64 = idx.to(torch::kInt64);
  auto expected = torch::take_along_dim(t, idx64, 0);

  auto hw_t = host_to_device<int32_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  // allocate output of shape {idx.size(0)}
  auto hw_out = empty<int32_t>({ idx.size(0) });

  take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out);
  auto out = device_to_host<int32_t>(hw_out);

  // Upcast out to int64 for comparison with expected
  ASSERT_TRUE(torch::equal(out.to(torch::kInt64), expected));
}

// 2D along axis 0
TEST(TakeAlongAxisTests, TwoDim_Axis0) {
    auto t = torch::arange(12, torch::kInt64).reshape({3,4});
    auto idx = torch::tensor({
      {0,1,2,0},
      {2,0,1,2},
      {1,2,0,1}
    }, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 0);

    auto hw_t = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({idx.size(0), idx.size(1)});

    take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out);
    auto out = device_to_host<int64_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// Full‑identity on a 3D tensor
TEST(TakeAlongAxisTests, IdentityIndex_3D) {
    auto t = torch::randint(-5,5,{2,2,2}, torch::kInt64);
    auto idx = torch::zeros_like(t, torch::kInt64);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k)
          idx.index_put_({i,j,k}, k);

    auto hw_t = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({idx.size(0), idx.size(1), idx.size(2)});

    take_along_axis<int64_t>(hw_t, hw_idx, 2, hw_out);
    auto out = device_to_host<int64_t>(hw_out);
    ASSERT_TRUE(torch::equal(t, out));
}

// Negative indices mapping
TEST(TakeAlongAxisTests, NegativeIndicesMapping) {
  auto t = torch::tensor({100,200,300}, torch::kInt64);
  auto idx = torch::tensor({-1,0,-2}, torch::kInt64);
  auto expected = torch::tensor({300,100,200}, torch::kInt64);

  auto hw_t = host_to_device<int64_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = empty<int64_t>({ idx.size(0) });

  take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out);
  auto out = device_to_host<int64_t>(hw_out);
  ASSERT_TRUE(torch::equal(out, expected));
}

// Negative axis for last dimension
TEST(TakeAlongAxisTests, NegativeAxisAsLastDim) {
  auto t = torch::arange(6, torch::kInt64).reshape({2,3});
  auto idx = torch::tensor({{2,1,0},{0,1,2}}, torch::kInt64);
  auto expected = torch::take_along_dim(t, idx, 1);

  auto hw_t = host_to_device<int64_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = empty<int64_t>({idx.size(0), idx.size(1)});

  take_along_axis<int64_t>(hw_t, hw_idx, -1, hw_out);
  auto out = device_to_host<int64_t>(hw_out);
  ASSERT_TRUE(torch::equal(out, expected));
}

// Non‑contiguous input (e.g. transpose) must still work
TEST(TakeAlongAxisTests, Works_OnNonContiguousInput) {
  auto base = torch::arange(6, torch::kInt64).reshape({2,3});
  auto t = base.transpose(0,1);
  auto idx = torch::tensor({{1,0},{0,1},{1,1}}, torch::kInt64);
  auto expected = torch::take_along_dim(t, idx, 1);

  auto hw_t = host_to_device<int64_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = empty<int64_t>({idx.size(0), idx.size(1)});

  take_along_axis<int64_t>(hw_t, hw_idx, 1, hw_out);
  auto out = device_to_host<int64_t>(hw_out);
  ASSERT_TRUE(torch::equal(out, expected));
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

 // Scalar input + scalar idx → should throw out_of_range (torch errors on rank 0)
TEST(TakeAlongAxisTests, ScalarInputAndScalarIdx_Throws) {
  auto t = torch::tensor(42, torch::kInt64);
  auto idx = torch::tensor(0,  torch::kInt64);

  auto hw_t = host_to_device<int64_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = empty<int64_t>({});  // shape = {}

  EXPECT_THROW(
    take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out),
    std::out_of_range
  );
}

// Rank‑mismatch between input and idx
TEST(TakeAlongAxisTests, Throws_OnRankMismatch) {
    auto t = torch::arange(4, torch::kInt64);       // rank=1
    auto idx = torch::randint(0,4,{2,2}, torch::kInt64);// rank=2
    auto hw_t = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({idx.size(0), idx.size(1)});

    EXPECT_THROW(
      take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out),
      std::invalid_argument
    );
}

// Axis too large or too negative for a multi‑dim tensor
TEST(TakeAlongAxisTests, Throws_OnAxisOutOfRange_MultiDim) {
    auto t = torch::randint(0,10,{3,3}, torch::kInt64);
    auto idx = torch::randint(0,3, {3,3}, torch::kInt64);
    auto hw_t = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({idx.size(0), idx.size(1)});

    EXPECT_THROW(
      take_along_axis<int64_t>(hw_t, hw_idx, 2, hw_out),
      std::out_of_range
    );
    EXPECT_THROW(
      take_along_axis<int64_t>(hw_t, hw_idx, -3, hw_out),
      std::out_of_range
    );
}

TEST(TakeAlongAxisTests, Throws_OnShapeMismatch) {
  auto t = torch::randint(-10, 10, {2,3,4,5}, torch::kInt64);
  // choose axis=2, so idx.shape = {2,3,6,5}
  auto idx = torch::randint(0, 4, {2,3,6,5}, torch::kInt64);

  auto expected = torch::take_along_dim(t, idx, 2);

  auto hw_t = host_to_device<int64_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = empty<int64_t>({idx.size(0), idx.size(1), idx.size(2), idx.size(3)});

  EXPECT_THROW(
    take_along_axis<int64_t>(hw_t, hw_idx, -3, hw_out),
    std::invalid_argument
  );
}
