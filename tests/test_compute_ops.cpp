#include "test_helpers.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            TAKE_ALONG_AXIS TESTS                               ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

// 1D gather along axis 0
TEST(TakeAlongAxisTests, Basic1D) {
    auto t   = torch::tensor({5, 10, 15, 20}, torch::kInt32);
    auto idx = torch::tensor({2, 0, 3, 1},     torch::kInt64);
    auto expected = torch::take_along_dim(t, idx, 0);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    // allocate output of shape {idx.size(0)}
    auto hw_out = empty<int32_t>({ idx.size(0) });

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
    auto hw_idx  = host_to_device<int64_t>(idx);
    // allocate output of shape {idx.size(0), idx.size(1)}
    auto hw_out  = empty<int64_t>({
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
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<float>({
      idx.size(0), idx.size(1)
    });

    take_along_axis<float>(hw_t, hw_idx, 1, hw_out);
    auto out = device_to_host<float>(hw_out);
    ASSERT_TRUE(torch::allclose(out, expected));
}

// Full‑identity on a 3D tensor
TEST(TakeAlongAxisTests, IdentityIndex_3D) {
    // manually specify input tensor
    auto t = torch::tensor({
        {{-1, 2},
        {3, -4}},
        {{5, -6},
        {-7, 8}}
    }, torch::kInt32);  // shape = (2,2,2)

    // manually specify the index tensor
    auto idx = torch::tensor({
        {{0, 1},
        {0, 1}},
        {{0, 1},
        {0, 1}}
    }, torch::kInt64);  // same shape, indexing along last dim=2

    // expected output using take_along_dim
    auto expected = torch::take_along_dim(t, idx, /*dim=*/2);

    // upload to device
    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int32_t>({
        idx.size(0), idx.size(1), idx.size(2)
    });

    // perform operation
    take_along_axis<int32_t>(hw_t, hw_idx, /*axis=*/2, hw_out);
    auto out = device_to_host<int32_t>(hw_out);

    ASSERT_TRUE(torch::equal(out, expected));
}


// 4‑D gather along axis 2 with fixed indices
TEST(TakeAlongAxisTests, FourDim_Axis2_Fixed) {
  // manually specify input tensor
  auto t = torch::tensor({
      { // batch 0
          { {  1,  2,  3,  4,  5 },   // 0th along axis=2
            {  6,  7,  8,  9, 10 },   // 1st
            { 11, 12, 13, 14, 15 },   // 2nd
            { 16, 17, 18, 19, 20 } }, // 3rd

          { { 21, 22, 23, 24, 25 },
            { 26, 27, 28, 29, 30 },
            { 31, 32, 33, 34, 35 },
            { 36, 37, 38, 39, 40 } },

          { { 41, 42, 43, 44, 45 },
            { 46, 47, 48, 49, 50 },
            { 51, 52, 53, 54, 55 },
            { 56, 57, 58, 59, 60 } }
      },
      { // batch 1
          { { 61, 62, 63, 64, 65 },
            { 66, 67, 68, 69, 70 },
            { 71, 72, 73, 74, 75 },
            { 76, 77, 78, 79, 80 } },

          { { 81, 82, 83, 84, 85 },
            { 86, 87, 88, 89, 90 },
            { 91, 92, 93, 94, 95 },
            { 96, 97, 98, 99,100 } },

          { {101,102,103,104,105},
            {106,107,108,109,110},
            {111,112,113,114,115},
            {116,117,118,119,120} }
      }
  }, torch::kInt32); // shape {2,3,4,5}

  // specify indices: idx.shape = {2,3,6,5}
  // pick indices 0,1,2,3 (valid for axis=2 size=4)
  auto idx = torch::tensor({
      { // batch 0
          { {0,1,2,3,0}, {1,2,3,0,1}, {2,3,0,1,2}, {3,0,1,2,3}, {0,1,2,3,0}, {1,2,3,0,1} },
          { {2,1,0,3,2}, {3,2,1,0,3}, {0,3,2,1,0}, {1,0,3,2,1}, {2,1,0,3,2}, {3,2,1,0,3} },
          { {1,2,3,0,1}, {2,3,0,1,2}, {3,0,1,2,3}, {0,1,2,3,0}, {1,2,3,0,1}, {2,3,0,1,2} }
      },
      { // batch 1
          { {3,2,1,0,3}, {2,1,0,3,2}, {1,0,3,2,1}, {0,3,2,1,0}, {3,2,1,0,3}, {2,1,0,3,2} },
          { {0,1,2,3,0}, {1,2,3,0,1}, {2,3,0,1,2}, {3,0,1,2,3}, {0,1,2,3,0}, {1,2,3,0,1} },
          { {2,1,0,3,2}, {3,2,1,0,3}, {0,3,2,1,0}, {1,0,3,2,1}, {2,1,0,3,2}, {3,2,1,0,3} }
      }
  }, torch::kInt64);

  auto expected = torch::take_along_dim(t, idx, /*dim=*/2);

  auto hw_t   = host_to_device<int32_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = empty<int32_t>({
    idx.size(0), idx.size(1), idx.size(2), idx.size(3)
  });

  take_along_axis<int32_t>(hw_t, hw_idx, /*axis=*/2, hw_out);
  auto out = device_to_host<int32_t>(hw_out);

  ASSERT_TRUE(torch::equal(out, expected));
}


// 4-D identity index along axis 0
TEST(TakeAlongAxisTests, IdentityGather4D_Axis0) {
    // shape = {3,2,3,2}, values = 0,1,2,…,35
    auto t = torch::arange(
        /*start=*/0,
        /*end=*/3*2*3*2,
        torch::kInt64
    ).reshape({3,2,3,2});

    // idx picks itself along dim=0: idx.shape = same as t
    auto idx = torch::empty_like(t, torch::kInt64);
    for (int i0 = 0; i0 < 3; ++i0)
    for (int i1 = 0; i1 < 2; ++i1)
    for (int i2 = 0; i2 < 3; ++i2)
    for (int i3 = 0; i3 < 2; ++i3) {
      idx.index_put_({i0,i1,i2,i3}, i0);
    }

    // since idx is identity along axis 0, expected == t
    auto expected = torch::take_along_dim(t, idx, /*axis=*/0);

    auto hw_t   = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({
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

  // first calculate number of elements
  int64_t numel = 1;
  for (auto d : a_shape) {
      numel *= d;
  }

  // then call arange properly
  auto t = torch::arange(0, numel, torch::TensorOptions().dtype(torch::kInt32))
              .reshape(a_shape);


  // manually create indices tensor
  auto idx = torch::empty(idx_shape, torch::kInt64);

  // simple predictable indices: cycle through 0,1,2,3,4
  for (int64_t i = 0; i < idx.numel(); ++i) {
      idx.view(-1)[i] = i % a_shape[7];  // valid indices into dim=7 (size 5)
  }

  // expected result: gather along dim=7
  auto expected = torch::take_along_dim(t, idx, /*dim=*/7);

  // upload
  auto hw_t   = host_to_device<int32_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);

  // allocate output with exactly the idx shape
  auto hw_out = empty<int32_t>({
      1,2,3,2,3,2,3,3,2,3,2
  });

  // run and compare
  take_along_axis<int32_t>(hw_t, hw_idx, /*axis=*/7, hw_out);
  auto out = device_to_host<int32_t>(hw_out);

  ASSERT_TRUE(torch::equal(out, expected));
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
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int32_t>({ idx.size(0) });

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
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({
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
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int32_t>({
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
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({
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
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = zeros<int32_t>({ idx.size(0) });

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
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<double>({
      idx.size(0), idx.size(1)
    });

    take_along_axis<double>(hw_t, hw_idx, 1, hw_out);
    auto out = device_to_host<double>(hw_out);
    ASSERT_TRUE(torch::allclose(out, expected));
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

// Scalar input + scalar idx → should throw out_of_range (torch errors on rank 0)
VALIDATION_TEST(TakeAlongAxisTests, ScalarInputAndScalarint64_throws) {
  auto t   = torch::tensor(42, torch::kInt32);
  auto idx = torch::tensor(0,  torch::kInt64);

  auto hw_t   = host_to_device<int32_t>(t);
  auto hw_idx = host_to_device<int64_t>(idx);
  auto hw_out = zeros<int32_t>({});  // shape = {}

  EXPECT_THROW(
    take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out),
    std::out_of_range
  );
}

// Mismatch in a *non*-axis dimension should throw
VALIDATION_TEST(TakeAlongAxisTests, Throws_OnShapeMismatchNonAxis) {
    auto t   = torch::randint(0, 10, {2, 3}, torch::kInt32);
    // dim 0 is 2 in `t`, but 1 in `idx`
    auto idx = torch::randint(0, 2, {1, 3}, torch::kInt64);

    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = zeros<int32_t>({ idx.size(0), idx.size(1) });

    EXPECT_THROW(
      take_along_axis<int32_t>(hw_t, hw_idx, /*axis=*/1, hw_out),
      std::invalid_argument
    );
}


// Rank‑mismatch between input and idx
VALIDATION_TEST(TakeAlongAxisTests, Throws_OnRankMismatch) {
    auto t   = torch::arange(4, torch::kInt32);       // rank=1
    auto idx = torch::randint(0,4,{2,2}, torch::kInt64);// rank=2
    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int32_t>({
      idx.size(0), idx.size(1)
    });

    EXPECT_THROW(
      take_along_axis<int32_t>(hw_t, hw_idx, 0, hw_out),
      std::invalid_argument
    );
}

// Axis too large or too negative for a multi‑dim tensor
VALIDATION_TEST(TakeAlongAxisTests, Throws_OnAxisOutOfRange_MultiDim) {
    auto t   = torch::randint(0,10,{3,3}, torch::kInt64);
    auto idx = torch::randint(0,3, {3,3}, torch::kInt64);
    auto hw_t   = host_to_device<int64_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int64_t>({
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
VALIDATION_TEST(TakeAlongAxisTests, Throws_OnAxisOutOfRange_ZeroDim) {
    auto t   = torch::tensor(7, torch::kInt32);
    auto idx = torch::tensor(0, torch::kInt64);
    auto hw_t   = host_to_device<int32_t>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<int32_t>({});  // scalar

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
VALIDATION_TEST(TakeAlongAxisTests, Throws_OnIndexOutOfBounds) {
    auto t   = torch::randn({4}, torch::kFloat);
    auto idx = torch::tensor({0,4,1}, torch::kInt64);
    auto hw_t   = host_to_device<float>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<float>({ idx.size(0) });

    EXPECT_THROW(
      take_along_axis<float>(hw_t, hw_idx, 0, hw_out),
      std::out_of_range
    );
}

// Broadcastable → mismatch on non‑axis should throw
VALIDATION_TEST(TakeAlongAxisTests, Throws_OnBroadcastableIdx) {
    auto t   = torch::randn({2,3,4}, torch::kFloat);
    // idx is {2,1,4}; non‑axis dim 1 is 1 vs input’s 3
    auto idx = torch::randint(0, 4, {2, 1, 4}, torch::kInt64);

    auto hw_t   = host_to_device<float>(t);
    auto hw_idx = host_to_device<int64_t>(idx);
    auto hw_out = empty<float>({ idx.size(0), idx.size(1), idx.size(2) });

    EXPECT_THROW(
      take_along_axis<float>(hw_t, hw_idx, /*axis=*/2, hw_out),
      std::invalid_argument
    );
}

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            APPLY_G_DECOMP TESTS                                ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

// Basic 1D integer decomposition
TEST(ApplyGDecompTests, Basic1D) {
  auto t_cpu   = torch::tensor({5,6,7}, torch::kInt32);
  int32_t g_exp       = 2;
  int32_t g_base_bits = 1;  // base = 2
  auto expected = torch::tensor({{1,0},{0,1},{1,1}}, torch::kInt32);

  auto hw_t   = host_to_device<int32_t>(t_cpu);
  auto hw_out = zeros<int32_t>({3,2});
  apply_g_decomp<int32_t>(hw_t, g_exp, g_base_bits, hw_out);
  auto out    = device_to_host<int32_t>(hw_out);

  ASSERT_TRUE(torch::equal(out, expected));
}

// Scalar input decomposition
TEST(ApplyGDecompTests, ScalarInput) {
  auto t_cpu   = torch::tensor(42, torch::kInt64);
  int32_t g_exp       = 3;
  int32_t g_base_bits = 3;  // base = 8
  auto expected = torch::tensor({2,5,0}, torch::kInt64);

  auto hw_t   = host_to_device<int64_t>(t_cpu);
  auto hw_out = zeros<int64_t>({3});
  apply_g_decomp<int64_t>(hw_t, g_exp, g_base_bits, hw_out);
  auto out    = device_to_host<int64_t>(hw_out);

  ASSERT_TRUE(torch::equal(out, expected));
}

// 2D test with base-4 decomposition
TEST(ApplyGDecompTests, TwoDim_Base4) {
  auto t_cpu = torch::tensor({
      {13, 7, 0},
      { 1,15, 8}
  }, torch::kInt32);
  int32_t g_exp       = 2;
  int32_t g_base_bits = 2;  // base = 4

  auto expected = torch::tensor({
    {{1,3},{3,1},{0,0}},
    {{1,0},{3,3},{0,2}}
  }, torch::kInt32);

  auto hw_t   = host_to_device<int32_t>(t_cpu);
  auto hw_out = zeros<int32_t>({2,3,g_exp});
  apply_g_decomp<int32_t>(hw_t, g_exp, g_base_bits, hw_out);
  auto out    = device_to_host<int32_t>(hw_out);

  ASSERT_TRUE(torch::equal(out, expected));
}

// 3D test with binary decomposition
TEST(ApplyGDecompTests, ThreeDim_Binary) {
  auto t_cpu = torch::tensor({
    {{2,5}},
    {{7,3}}
  }, torch::kInt64);
  int32_t g_exp       = 3;
  int32_t g_base_bits = 1;  // base = 2

  auto expected = torch::tensor({
    {{{0,1,0},{1,0,1}}},
    {{{1,1,1},{1,1,0}}}
  }, torch::kInt64);

  auto hw_t   = host_to_device<int64_t>(t_cpu);
  auto hw_out = zeros<int64_t>({2,1,2,g_exp});
  apply_g_decomp<int64_t>(hw_t, g_exp, g_base_bits, hw_out);
  auto out    = device_to_host<int64_t>(hw_out);

  ASSERT_TRUE(torch::equal(out, expected));
}

// Full bit-width reconstruction test
TEST(ApplyGDecompTests, FullBitWidthReconstruction) {
  // force each hex literal into a signed 32-bit int
  auto t_cpu = torch::tensor(std::vector<int32_t>{
      static_cast<int32_t>(0x12345678),
      static_cast<int32_t>(0xDEADBEEF),
      static_cast<int32_t>(0x0F0F0F0F),
      static_cast<int32_t>(0x80000000)
  }, torch::kInt32);

  int32_t g_exp       = 8;
  int32_t g_base_bits = 4;  // base = 16, covers 32 bits

  auto hw_t   = host_to_device<int32_t>(t_cpu);
  auto hw_out = zeros<int32_t>({4, g_exp});
  apply_g_decomp<int32_t>(hw_t, g_exp, g_base_bits, hw_out);
  auto out    = device_to_host<int32_t>(hw_out);

  for (int i = 0; i < 4; ++i) {
      int32_t orig = t_cpu[i].item<int32_t>();
      int64_t rec = 0;
      for (int j = 0; j < g_exp; ++j) {
          int32_t d = out.index({i,j}).item<int32_t>();
          rec |= (int64_t(d) << (j * g_base_bits));
      }
      ASSERT_EQ(static_cast<int32_t>(rec), orig)
          << "reconstruction failed at index " << i;
  }
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

// Empty input produces empty output of correct shape
TEST(ApplyGDecompTests, EmptyInputProducesEmpty) {
  auto t_cpu   = torch::empty({0,5}, torch::kInt32);
  int32_t g_exp       = 3;
  int32_t g_base_bits = 2;
  auto hw_t   = host_to_device<int32_t>(t_cpu);
  auto hw_out = zeros<int32_t>({0,5,g_exp});
  apply_g_decomp<int32_t>(hw_t, g_exp, g_base_bits, hw_out);
  auto out    = device_to_host<int32_t>(hw_out);

  ASSERT_EQ(out.numel(), 0);
  ASSERT_EQ(out.sizes(), (std::vector<int64_t>{0,5,g_exp}));
}

// Negative inputs get their two's-complement low bits
TEST(ApplyGDecompTests, NegativeValuesBinary) {
  // -5 = …11111011₂ → bits [1,1,0,1]  (LSB first)
  // -2 = …11111110₂ → bits [0,1,1,1]
  auto t_cpu   = torch::tensor({-5, -2}, torch::kInt32);
  int32_t g_exp       = 4;
  int32_t g_base_bits = 1;  // binary
  auto expected = torch::tensor({{1,1,0,1},{0,1,1,1}}, torch::kInt32);

  auto hw_t    = host_to_device<int32_t>(t_cpu);
  auto hw_out  = zeros<int32_t>({2, g_exp});
  apply_g_decomp<int32_t>(hw_t, g_exp, g_base_bits, hw_out);
  auto out     = device_to_host<int32_t>(hw_out);

  ASSERT_TRUE(torch::equal(out, expected));
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

// Parameter validation: non-positive g_exp
VALIDATION_TEST(ApplyGDecompTests, Throws_OnNonPositiveGExp) {
  auto t      = host_to_device<int32_t>(torch::tensor({3,5}, torch::kInt32));
  auto result = zeros<int32_t>({2,2});
  EXPECT_THROW(apply_g_decomp<int32_t>(t, /*g_exp=*/0, /*g_base_bits=*/1, result),
               std::invalid_argument);
  EXPECT_THROW(apply_g_decomp<int32_t>(t, /*g_exp=*/-1, /*g_base_bits=*/1, result),
               std::invalid_argument);
}

// Parameter validation: non-positive g_base_bits
VALIDATION_TEST(ApplyGDecompTests, Throws_OnNonPositiveBaseBits) {
  auto t      = host_to_device<int64_t>(torch::tensor({7,8}, torch::kInt64));
  auto result = zeros<int64_t>({2,2});
  EXPECT_THROW(apply_g_decomp<int64_t>(t, /*g_exp=*/2, /*g_base_bits=*/0, result),
               std::invalid_argument);
  EXPECT_THROW(apply_g_decomp<int64_t>(t, /*g_exp=*/2, /*g_base_bits=*/-3, result),
               std::invalid_argument);
}

// Parameter validation: g_base_bits too large for type
VALIDATION_TEST(ApplyGDecompTests, Throws_OnBaseBitsTooLarge) {
  auto t32   = host_to_device<int32_t>(torch::tensor({1,2,3}, torch::kInt32));
  auto bad32 = zeros<int32_t>({3,33});
  EXPECT_THROW(apply_g_decomp<int32_t>(t32, /*g_exp=*/33, /*g_base_bits=*/33, bad32),
               std::invalid_argument);

  auto t64   = host_to_device<int64_t>(torch::tensor({1,2}, torch::kInt64));
  auto bad64 = zeros<int64_t>({2,65});
  EXPECT_THROW(apply_g_decomp<int64_t>(t64, /*g_exp=*/65, /*g_base_bits=*/65, bad64),
               std::invalid_argument);
}

// Shape validation: result rank must be input rank + 1
VALIDATION_TEST(ApplyGDecompTests, Throws_OnWrongResultRank) {
  auto t   = host_to_device<int32_t>(torch::tensor({1,2,3}, torch::kInt32));
  auto bad = zeros<int32_t>({3});  // rank=1 instead of 2
  EXPECT_THROW(apply_g_decomp<int32_t>(t, /*g_exp=*/2, /*g_base_bits=*/1, bad),
               std::invalid_argument);
}

// Shape validation: trailing dim must equal g_exp
VALIDATION_TEST(ApplyGDecompTests, Throws_OnTrailingDimNotEqualGExp) {
  auto t   = host_to_device<int64_t>(torch::tensor({4,5}, torch::kInt64));
  auto bad = zeros<int64_t>({2,3});  // g_exp=2, trailing size=3
  EXPECT_THROW(apply_g_decomp<int64_t>(t, /*g_exp=*/2, /*g_base_bits=*/1, bad),
               std::invalid_argument);
}

// Shape validation: non-trailing dims must match input dims
VALIDATION_TEST(ApplyGDecompTests, Throws_OnDimMismatch) {
  auto t   = host_to_device<int32_t>(torch::tensor({1,2,3}, torch::kInt32));
  auto bad = zeros<int32_t>({2,2});  // first dim 2 vs 3
  EXPECT_THROW(apply_g_decomp<int32_t>(t, /*g_exp=*/2, /*g_base_bits=*/1, bad),
               std::invalid_argument);
}


/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                              ABS FUNCTION TESTS                                ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

// 1D integer abs
TEST(AbsTests, Basic1DInt) {
  auto t_cpu   = torch::tensor({-5, 0, 7, -2}, torch::kInt32);
  auto expected = torch::tensor({5, 0, 7, 2}, torch::kInt32);

  auto hw_t   = host_to_device<int32_t>(t_cpu);
  auto hw_out = zeros<int32_t>({4});
  abs<int32_t>(hw_t, hw_out);
  auto out    = device_to_host<int32_t>(hw_out);

  ASSERT_TRUE(torch::equal(out, expected));
}

// 1D floating‐point abs
TEST(AbsTests, Basic1DFloat) {
  auto t_cpu   = torch::tensor({-3.5f, 0.0f, 2.25f, -0.125f}, torch::kFloat32);
  auto expected = torch::tensor({3.5f, 0.0f, 2.25f, 0.125f}, torch::kFloat32);

  auto hw_t   = host_to_device<float>(t_cpu);
  auto hw_out = zeros<float>({4});
  abs<float>(hw_t, hw_out);
  auto out    = device_to_host<float>(hw_out);

  ASSERT_TRUE(torch::allclose(out, expected));
}

// Scalar integer input
TEST(AbsTests, ScalarInt) {
  auto t_cpu   = torch::tensor(-42, torch::kInt64);
  auto expected = torch::tensor(42, torch::kInt64);

  auto hw_t   = host_to_device<int64_t>(t_cpu);
  auto hw_out = zeros<int64_t>({});
  abs<int64_t>(hw_t, hw_out);
  auto out    = device_to_host<int64_t>(hw_out);

  ASSERT_EQ(out.item<int64_t>(), expected.item<int64_t>());
}

// 2D mixed‐sign double
TEST(AbsTests, TwoDDouble) {
  auto t_cpu = torch::tensor({
      {-1.0,  2.5,  0.0},
      { 3.141, -4.2, -0.001}
  }, torch::kFloat64);
  auto expected = torch::tensor({
      {1.0,  2.5,   0.0},
      {3.141, 4.2,  0.001}
  }, torch::kFloat64);

  auto hw_t   = host_to_device<double>(t_cpu);
  auto hw_out = zeros<double>({2,3});
  abs<double>(hw_t, hw_out);
  auto out    = device_to_host<double>(hw_out);

  ASSERT_TRUE(torch::allclose(out, expected, /*rtol=*/1e-7, /*atol=*/1e-8));
}

/************************************************************************************************
* Edge & corner cases
***********************************************************************************************/

// Empty tensor yields empty result of same shape
TEST(AbsTests, EmptyInputProducesEmpty) {
  auto t_cpu   = torch::empty({0,5}, torch::kInt32);
  auto hw_t    = host_to_device<int32_t>(t_cpu);
  auto hw_out  = zeros<int32_t>({0,5});
  abs<int32_t>(hw_t, hw_out);
  auto out     = device_to_host<int32_t>(hw_out);

  ASSERT_EQ(out.numel(), 0);
  ASSERT_EQ(out.sizes(), (std::vector<int64_t>{0,5}));
}

/************************************************************************************************
* Error conditions
***********************************************************************************************/

// Shape mismatch: dims must match exactly
VALIDATION_TEST(AbsTests, Throws_OnShapeMismatch) {
  auto t    = host_to_device<int32_t>(torch::tensor({1,2,3}, torch::kInt32));
  auto bad  = zeros<int32_t>({2});  // wrong size
  EXPECT_THROW(abs<int32_t>(t, bad), std::invalid_argument);
}

/***************************************************************************************
****************************************************************************************
****                                                                                ****
****                        SET_CONST_VAL FUNCTION TESTS                            ****
****                                                                                ****
****************************************************************************************
****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

TEST(SetConstValTests, Basic1DInt32) {
  // start with some arbitrary data
  auto t_cpu    = torch::tensor({1, 2, -3, 42}, torch::kInt32);
  auto hw_t     = host_to_device<int32_t>(t_cpu);

  // set every element to 7
  set_const_val<int32_t>(hw_t, 7);
  auto out      = device_to_host<int32_t>(hw_t);
  auto expected = torch::full({4}, 7, torch::kInt32);

  ASSERT_TRUE(torch::equal(out, expected));
}

TEST(SetConstValTests, Basic1DFloat) {
  auto t_cpu    = torch::tensor({0.1f, -2.5f, 3.14f}, torch::kFloat32);
  auto hw_t     = host_to_device<float>(t_cpu);

  set_const_val<float>(hw_t, -1.25f);
  auto out      = device_to_host<float>(hw_t);
  auto expected = torch::full({3}, -1.25f, torch::kFloat32);

  ASSERT_TRUE(torch::allclose(out, expected));
}

TEST(SetConstValTests, ScalarInt64) {
  // zero‐dimensional tensor
  auto t_cpu = torch::tensor(123, torch::kInt64);
  auto hw_t  = host_to_device<int64_t>(t_cpu);

  set_const_val<int64_t>(hw_t, -999LL);
  auto out   = device_to_host<int64_t>(hw_t);

  ASSERT_EQ(out.item<int64_t>(), -999LL);
}

TEST(SetConstValTests, TwoDDouble) {
  // 2×3 tensor, random initial contents
  auto t_cpu    = torch::rand({2, 3}, torch::kFloat64);
  auto hw_t     = host_to_device<double>(t_cpu);

  set_const_val<double>(hw_t, 0.0);
  auto out      = device_to_host<double>(hw_t);
  auto expected = torch::zeros({2, 3}, torch::kFloat64);

  ASSERT_TRUE(torch::allclose(out, expected, /*rtol=*/1e-7, /*atol=*/1e-8));
}

/************************************************************************************************
* Edge & corner cases
***********************************************************************************************/

TEST(SetConstValTests, EmptyTensor) {
  // shape {0,5} → zero elements
  auto t_cpu = torch::empty({0, 5}, torch::kInt32);
  auto hw_t  = host_to_device<int32_t>(t_cpu);

  set_const_val<int32_t>(hw_t, 42);
  auto out   = device_to_host<int32_t>(hw_t);

  ASSERT_EQ(out.numel(), 0);
  ASSERT_EQ(out.sizes(), (std::vector<int64_t>{0, 5}));
}

TEST(SetConstValTests, LargeTensor) {
  // stress test: 100K elements
  const int N = 100000;
  auto t_cpu    = torch::arange(0, N, torch::kInt32);
  auto hw_t     = host_to_device<int32_t>(t_cpu);

  set_const_val<int32_t>(hw_t, 123);
  auto out      = device_to_host<int32_t>(hw_t);
  auto expected = torch::full({N}, 123, torch::kInt32);

  ASSERT_TRUE(torch::equal(out, expected));
}

/************************************************************************************************
* Error conditions
***********************************************************************************************/

VALIDATION_TEST(SetConstValTests, Throws_OnNullTensor) {
  std::shared_ptr<DeviceTensor<int32_t>> null_ptr;
  EXPECT_THROW(
    set_const_val<int32_t>(null_ptr, 5),
    std::invalid_argument
  );
}