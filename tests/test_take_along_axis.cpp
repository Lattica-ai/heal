#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

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

// 4‑D gather along axis 2 (random data)
TEST(TakeAlongAxisTests, FourDim_Axis2_RepeatedIndices) {
    auto t = torch::randint(-10, 10, {2,3,4,5}, torch::kInt64);
    // choose axis=2, so idx.shape = {2,3,6,5}
    auto idx = torch::randint(0, 4, {2,3,6,5}, torch::kInt64);

    auto expected = torch::take_along_dim(t, idx, 2);

    auto hw_t = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({idx.size(0), idx.size(1), idx.size(2), idx.size(3)});

    take_along_axis<int64_t>(hw_t, hw_idx, 2, hw_out);
    auto out = device_to_host<int64_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// 4-D gather along axis 2, but asking for fewer values than exist
TEST(TakeAlongAxisTests, FourDim_Axis2_LessIndices) {
  auto t = torch::randint(-10, 10, {2,3,4,5}, torch::kInt64);

  // we gather only 2 elements along dim-2, so idx.shape = [2,3,2,5]
  auto idx = torch::randint(0, 4, {2,3,2,5}, torch::kInt64);

  auto expected = torch::take_along_dim(t, idx, 2);

  auto hw_t = host_to_device<int64_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = empty<int64_t>({idx.size(0), idx.size(1), idx.size(2), idx.size(3)});

  take_along_axis<int64_t>(hw_t, hw_idx, 2, hw_out);
  auto out = device_to_host<int64_t>(hw_out);

  // should match, and out.shape == [2,3,2,5]
  ASSERT_TRUE(torch::equal(out, expected));
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

// Out‑of‑bounds indices along the gather axis
TEST(TakeAlongAxisTests, Throws_OnIndexOutOfBounds) {
  auto t = torch::arange(0, 4, torch::kInt64);
  auto idx = torch::tensor({0,4,1,2}, torch::kInt64);
    auto hw_t = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({ idx.size(0) });

    EXPECT_THROW(
      take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out),
      std::out_of_range
    );
}

// Out‑of‑bounds indices along the gather axis
TEST(TakeAlongAxisTests, Throws_OnNegIndexOutOfBounds) {
  auto t = torch::arange(0, 4, torch::kInt64);
  auto idx = torch::tensor({0,-5,1,2}, torch::kInt64);
    auto hw_t = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({ idx.size(0) });

    EXPECT_THROW(
      take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out),
      std::out_of_range
    );
}

// Broadcastable (but unsupported) → mismatch on non‑axis should throw
TEST(TakeAlongAxisTests, Throws_OnBroadcastableIdx) {
  auto t = torch::arange(2 * 3 * 4, torch::kInt64).reshape({2, 3, 4});
  // idx is {2, 1, 4}; non‑axis dim 1 is 1 vs input’s 3
  auto idx = torch::randint(0, 4, {2, 1, 4}, torch::kInt64);

  auto hw_t = host_to_device<int64_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = empty<int64_t>({ idx.size(0), idx.size(1), idx.size(2) });

  EXPECT_THROW(
      take_along_axis<int64_t>(hw_t, hw_idx, 2, hw_out),
      std::invalid_argument
  );
}