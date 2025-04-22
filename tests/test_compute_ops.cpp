#include "gtest/gtest.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;
typedef IndexType idx_t;

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

// 1D gather along axis 0
TEST(TakeAlongAxisTests, Basic1D) {
    auto t   = torch::tensor({5, 10, 15, 20}, torch::kInt32);
    auto idx = torch::tensor({2, 0, 3, 1},     torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 0);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    // allocate output of shape {idx.size(0)}
    auto hw_out = allocate_on_hardware<int32_t>({ idx.size(0) });

    take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out);
    auto out = device_to_host<int32_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// 2D along axis 0
TEST(TakeAlongAxisTests, TwoDim_Axis0) {
    auto t   = torch::arange(12, torch::kInt64).reshape({3,4});
    auto idx = torch::tensor({
      {0,1,2,0},
      {2,0,1,2},
      {1,2,0,1}
    }, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 0);

    auto hw_t    = host_to_device<int64_t>(t);
    auto hw_idx  = host_to_device<idx_t>(idx);
    // allocate output of shape {idx.size(0), idx.size(1)}
    auto hw_out  = allocate_on_hardware<int64_t>({
      idx.size(0), idx.size(1)
    });

    take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out);
    auto out = device_to_host<int64_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// 2D float along axis 1
TEST(TakeAlongAxisTests, TwoDim_Axis1_Float) {
    auto t   = torch::arange(12, torch::kFloat).reshape({3,4});
    auto idx = torch::tensor({
      {3,2,1,0},
      {0,1,2,3},
      {1,1,1,1}
    }, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 1);

    auto hw_t   = host_to_device<float>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<float>({
      idx.size(0), idx.size(1)
    });

    take_along_axis<float>(hw_t, hw_idx, 1, hw_out);
    auto out = device_to_host<float>(hw_out);
    ASSERT_TRUE(torch::allclose(out, expected));
}

// Scalar input + scalar idx → should throw out_of_range (torch errors on rank 0)
TEST(TakeAlongAxisTests, ScalarInputAndScalarIdx_Throws) {
    auto t   = torch::tensor(42, torch::kInt32);
    auto idx = torch::tensor(0,  torch::kInt64);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({});  // shape = {}

    EXPECT_THROW(
      take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out),
      std::out_of_range
    );
}

// Full‑identity on a 3D tensor
TEST(TakeAlongAxisTests, IdentityIndex_3D) {
    auto t   = torch::randint(-5,5,{2,2,2}, torch::kInt32);
    auto idx = torch::zeros_like(t, torch::kInt64);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k)
          idx.index_put_({i,j,k}, k);

    auto expected = torch::take_along_dim(t, idx, 2);
    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({
      idx.size(0), idx.size(1), idx.size(2)
    });

    take_along_axis<int32_t>(hw_t, hw_idx, 2, hw_out);
    auto out = device_to_host<int32_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// 4‑D gather along axis 2 (random data)
TEST(TakeAlongAxisTests, FourDim_Axis2_Random) {
    // shape = {2,3,4,5}
    auto t = torch::randint(-10, 10, {2,3,4,5}, torch::kInt32);
    // choose axis=2, so idx.shape = {2,3,6,5}
    auto idx = torch::randint(0, 4, {2,3,6,5}, torch::kInt64);

    auto expected = torch::take_along_dim(t, idx, 2);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({
      idx.size(0), idx.size(1), idx.size(2), idx.size(3)
    });

    take_along_axis<int32_t>(hw_t, hw_idx, /*axis=*/2, hw_out);
    auto out = device_to_host<int32_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// 4‑D identity index along axis 0
TEST(TakeAlongAxisTests, FourDim_Axis0_Identity) {
    // shape = {3,2,3,2}
    auto t = torch::randint(0, 20, {3,2,3,2}, torch::kInt64);
    // idx picks itself along dim=0: idx.shape = same as t
    auto idx = torch::zeros_like(t, torch::kInt64);
    for (int i0 = 0; i0 < 3; ++i0)
    for (int i1 = 0; i1 < 2; ++i1)
    for (int i2 = 0; i2 < 3; ++i2)
    for (int i3 = 0; i3 < 2; ++i3) {
      idx.index_put_({i0,i1,i2,i3}, i0);
    }

    auto expected = torch::take_along_dim(t, idx, /*axis=*/0);

    auto hw_t   = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int64_t>({
      idx.size(0), idx.size(1), idx.size(2), idx.size(3)
    });

    take_along_axis<int64_t>(hw_t, hw_idx, 0, hw_out);
    auto out = device_to_host<int64_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// 11‑D gather along axis 7 with varied dimension sizes
TEST(TakeAlongAxisTests, ElevenDim_VariedShape_Axis7) {
    // input shape:  {1,2,3,2,3,2,3,5,2,3,2}
    // we’ll gather along dim=7, so idx_shape[7]=3
    auto a_shape = std::vector<int64_t>{1,2,3,2,3,2,3,5,2,3,2};
    auto idx_shape = a_shape;
    idx_shape[7] = 3;

    // random input and indices
    auto t   = torch::randint(-100, 100, a_shape,    torch::kInt32);
    auto idx = torch::randint(0, a_shape[7], idx_shape, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, /*dim=*/7);

    // upload
    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);

    // allocate output with exactly the idx shape
    auto hw_out = allocate_on_hardware<int32_t>({
        1,2,3,2,3,2,3,3,2,3,2
    });

    // run and compare
    take_along_axis<int32_t>(hw_t, hw_idx, /*axis=*/7, hw_out);
    auto out = device_to_host<int32_t>(hw_out);

    ASSERT_TRUE(torch::equal(out, expected));
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

// Mismatch in a *non*-axis dimension should throw
TEST(TakeAlongAxisTests, Throws_OnShapeMismatchNonAxis) {
    auto t   = torch::randint(0, 10, {2, 3}, torch::kInt32);
    // dim 0 is 2 in `t`, but 1 in `idx`
    auto idx = torch::randint(0, 2, {1, 3}, torch::kInt64);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({ idx.size(0), idx.size(1) });

    EXPECT_THROW(
      take_along_axis<int32_t>(hw_t, hw_idx, /*axis=*/1, hw_out),
      std::invalid_argument
    );
}


// Rank‑mismatch between input and idx
TEST(TakeAlongAxisTests, Throws_OnRankMismatch) {
    auto t   = torch::arange(4, torch::kInt32);       // rank=1
    auto idx = torch::randint(0,4,{2,2}, torch::kInt64);// rank=2
    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({
      idx.size(0), idx.size(1)
    });

    EXPECT_THROW(
      take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out),
      std::invalid_argument
    );
}

// Axis too large or too negative for a multi‑dim tensor
TEST(TakeAlongAxisTests, Throws_OnAxisOutOfRange_MultiDim) {
    auto t   = torch::randint(0,10,{3,3}, torch::kInt64);
    auto idx = torch::randint(0,3, {3,3}, torch::kInt64);
    auto hw_t   = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int64_t>({
      idx.size(0), idx.size(1)
    });

    EXPECT_THROW(
      take_along_axis<int64_t>(hw_t, hw_idx, 2, hw_out),
      std::out_of_range
    );
    EXPECT_THROW(
      take_along_axis<int64_t>(hw_t, hw_idx, -3, hw_out),
      std::out_of_range
    );
}

// Axis too large or too negative for a scalar
TEST(TakeAlongAxisTests, Throws_OnAxisOutOfRange_ZeroDim) {
    auto t   = torch::tensor(7, torch::kInt32);
    auto idx = torch::tensor(0, torch::kInt64);
    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({});  // scalar

    EXPECT_THROW(
      take_along_axis<int32_t>(hw_t, hw_idx, 1, hw_out),
      std::out_of_range
    );
    EXPECT_THROW(
      take_along_axis<int32_t>(hw_t, hw_idx, -2, hw_out),
      std::out_of_range
    );
}

// Out‑of‑bounds indices along the gather axis
TEST(TakeAlongAxisTests, Throws_OnIndexOutOfBounds) {
    auto t   = torch::randn({4}, torch::kFloat);
    auto idx = torch::tensor({0,4,1}, torch::kInt64);
    auto hw_t   = host_to_device<float>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<float>({ idx.size(0) });

    EXPECT_THROW(
      take_along_axis<float>(hw_t, hw_idx, 0, hw_out),
      std::out_of_range
    );
}

// Broadcastable (but unsupported) → mismatch on non‑axis should throw
TEST(TakeAlongAxisTests, Throws_OnBroadcastableIdx) {
    auto t   = torch::randn({2,3,4}, torch::kFloat);
    // idx is {2,1,4}; non‑axis dim 1 is 1 vs input’s 3
    auto idx = torch::randint(0, 4, {2, 1, 4}, torch::kInt64);

    auto hw_t   = host_to_device<float>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<float>({ idx.size(0), idx.size(1), idx.size(2) });

    EXPECT_THROW(
      take_along_axis<float>(hw_t, hw_idx, /*axis=*/2, hw_out),
      std::invalid_argument
    );
}


/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

// Negative indices mapping
TEST(TakeAlongAxisTests, NegativeIndicesMapping) {
    auto t   = torch::tensor({100,200,300}, torch::kInt32);
    auto idx = torch::tensor({-1,0,-2},     torch::kInt64);
    auto expected = torch::tensor({300,100,200}, torch::kInt32);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({ idx.size(0) });

    take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out);
    auto out = device_to_host<int32_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// Negative axis for last dimension
TEST(TakeAlongAxisTests, NegativeAxisAsLastDim) {
    auto t   = torch::arange(6, torch::kInt64).reshape({2,3});
    auto idx = torch::tensor({{2,1,0},{0,1,2}}, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 1);

    auto hw_t   = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int64_t>({
      idx.size(0), idx.size(1)
    });

    take_along_axis<int64_t>(hw_t, hw_idx, -1, hw_out);
    auto out = device_to_host<int64_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// Zero‑length dimensions → empty output of same shape
TEST(TakeAlongAxisTests, EmptyDimProducesEmpty) {
    auto t   = torch::randint(0,10,{0,5}, torch::kInt32);
    auto idx = torch::empty({0,5},   torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 0);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({
      idx.size(0), idx.size(1)
    });

    take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out);
    auto out = device_to_host<int32_t>(hw_out);
    ASSERT_EQ(out.numel(), 0);
    ASSERT_EQ(out.sizes(), expected.sizes());
}

// Non‑contiguous input (e.g. transpose) must still work
TEST(TakeAlongAxisTests, Works_OnNonContiguousInput) {
    auto base = torch::arange(6, torch::kInt64).reshape({2,3});
    auto t    = base.transpose(0,1);
    auto idx  = torch::tensor({{1,0},{0,1},{1,1}}, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 1);

    auto hw_t   = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int64_t>({
      idx.size(0), idx.size(1)
    });

    take_along_axis<int64_t>(hw_t, hw_idx, 1, hw_out);
    auto out = device_to_host<int64_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

// Large tensor stress test
TEST(TakeAlongAxisTests, LargeTensorStress) {
    const int N = 1 << 20;
    auto t   = torch::arange(N,   torch::kInt32);
    auto idx = torch::arange(0, N, 2, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 0);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<int32_t>({ idx.size(0) });

    take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out);
    auto out = device_to_host<int32_t>(hw_out);
    ASSERT_TRUE(torch::equal(out, expected));
}

/************************************************************************************************
 * Type‑coverage extras
 ***********************************************************************************************/

// Double precision
TEST(TakeAlongAxisTests, DoubleType) {
    auto t   = torch::arange(9, torch::kDouble).reshape({3,3});
    auto idx = torch::tensor({{2,1,0},{0,1,2},{1,1,1}}, torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 1);

    auto hw_t   = host_to_device<double>(t);
    auto hw_idx = host_to_device<idx_t>(idx);
    auto hw_out = allocate_on_hardware<double>({
      idx.size(0), idx.size(1)
    });

    take_along_axis<double>(hw_t, hw_idx, 1, hw_out);
    auto out = device_to_host<double>(hw_out);
    ASSERT_TRUE(torch::allclose(out, expected));
}
