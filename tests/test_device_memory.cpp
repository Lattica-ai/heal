#include "test_helpers.h"
#include "lattica_hw_api.h"
#include <torch/torch.h>

using namespace lattica_hw_api;


// Simulation flag for malloc failure
extern "C" {
    int malloc_fail_flag = 0;
    void* __real_malloc(size_t size);
    void* __wrap_malloc(size_t size) {
        if (malloc_fail_flag) {
            malloc_fail_flag = 0;
            return nullptr;
        }
        return __real_malloc(size);
    }
}


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            EMPTY TESTS                                         ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(EmptyTests, Basic1DInt64) {
    std::vector<int64_t> dims{4};
    auto device_tensor = empty<int64_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int64_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 4);
}

TEST(EmptyTests, Basic2DInt32) {
    std::vector<int64_t> dims{2, 3};
    auto device_tensor = empty<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 2 * 3);
}

TEST(EmptyTests, Basic1DFloat) {
    std::vector<int64_t> dims{5};
    auto device_tensor = empty<float>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<float>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 5);
}

TEST(EmptyTests, Basic3DDouble) {
    std::vector<int64_t> dims{3, 3, 3};
    auto device_tensor = empty<double>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<double>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 3 * 3 * 3);
    ASSERT_TRUE(torch_tensor.is_contiguous());
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(EmptyTests, EmptyTensor) {
    std::vector<int64_t> dims{0};
    auto device_tensor = empty<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 0);
}

TEST(EmptyTests, ZeroSizeDimension) {
    std::vector<int64_t> dims{0, 4};
    auto device_tensor = empty<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 0 * 4);
}

TEST(EmptyTests, LargeTensor) {
    std::vector<int64_t> dims{1000, 1000};
    auto device_tensor = empty<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 1000 * 1000);
}

/************************************************************************************************
 * Error conditions
 ***********************************************************************************************/

VALIDATION_TEST(EmptyTests, ThrowsOnNegativeDims) {
    EXPECT_THROW(empty<int32_t>({-1, 2, 3}), std::invalid_argument);
}

VALIDATION_TEST(EmptyTests, ThrowsOnAllocationFailure) {
    malloc_fail_flag = 1;
    EXPECT_THROW(empty<int32_t>({10, 10}), std::bad_alloc);
}

/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            ZEROS TESTS                                         ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(ZerosTests, Basic1DInt64) {
    std::vector<int64_t> dims{4};
    auto device_tensor = zeros<int64_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int64_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 4);
    // all elements should be zero
    ASSERT_TRUE(torch_tensor.eq(0).all().item<bool>());
}

TEST(ZerosTests, Basic2DInt32) {
    std::vector<int64_t> dims{2, 3};
    auto device_tensor = zeros<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 2 * 3);
    ASSERT_TRUE(torch_tensor.eq(0).all().item<bool>());
}

TEST(ZerosTests, Basic1DFloat) {
    std::vector<int64_t> dims{5};
    auto device_tensor = zeros<float>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<float>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 5);
    ASSERT_TRUE(torch_tensor.eq(0).all().item<bool>());
}

TEST(ZerosTests, Basic3DDouble) {
    std::vector<int64_t> dims{3, 3, 3};
    auto device_tensor = zeros<double>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<double>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 3 * 3 * 3);
    ASSERT_TRUE(torch_tensor.is_contiguous());
    ASSERT_TRUE(torch_tensor.eq(0).all().item<bool>());
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(ZerosTests, EmptyTensor) {
    std::vector<int64_t> dims{0};
    auto device_tensor = zeros<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 0);
    // an empty tensor trivially satisfies "all zero"
    ASSERT_TRUE(torch_tensor.eq(0).all().item<bool>());
}

TEST(ZerosTests, ZeroSizeDimension) {
    std::vector<int64_t> dims{0, 4};
    auto device_tensor = zeros<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 0 * 4);
    ASSERT_TRUE(torch_tensor.eq(0).all().item<bool>());
}

TEST(ZerosTests, LargeTensor) {
    std::vector<int64_t> dims{1000, 1000};
    auto device_tensor = zeros<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);
    torch::Tensor torch_tensor = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(torch_tensor.numel(), 1000 * 1000);
    ASSERT_TRUE(torch_tensor.eq(0).all().item<bool>());
}

/************************************************************************************************
 * Error conditions for zeros
 ***********************************************************************************************/

VALIDATION_TEST(ZerosTests, ThrowsOnNegativeDims) {
    EXPECT_THROW(zeros<int32_t>({-1, 2, 3}), std::invalid_argument);
}

VALIDATION_TEST(ZerosTests, ThrowsOnAllocationFailure) {
    malloc_fail_flag = 1;
    EXPECT_THROW(zeros<int32_t>({10, 10}), std::bad_alloc);
}


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            HOST_TO_DEVICE TESTS                                ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(HostToDeviceTests, Basic1DInt64) {
    // create a CPU tensor of shape [4], dtype int64, with values 0,1,2,3
    torch::Tensor t = torch::arange(0, 4, torch::TensorOptions().dtype(torch::kInt64));
    auto device_tensor = host_to_device<int64_t>(t);
    ASSERT_TRUE(device_tensor != nullptr);

    // round trip back to host
    torch::Tensor round = device_to_host<int64_t>(device_tensor);
    ASSERT_EQ(round.sizes(), t.sizes());
    // check contents
    ASSERT_TRUE(round.eq(t).all().item<bool>());
}

TEST(HostToDeviceTests, Basic2DFloat) {
    // shape [2,3], dtype float, random values
    torch::Tensor t = torch::rand({2,3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto device_tensor = host_to_device<float>(t);
    ASSERT_TRUE(device_tensor != nullptr);

    auto round = device_to_host<float>(device_tensor);
    ASSERT_EQ(round.numel(), 2 * 3);
    ASSERT_TRUE(round.allclose(t));
}

TEST(HostToDeviceTests, Basic3DDouble) {
    // shape [3,3,3], dtype double, fill with 7.5
    torch::Tensor t = torch::full({3,3,3}, 7.5, torch::TensorOptions().dtype(torch::kFloat64));
    auto device_tensor = host_to_device<double>(t);
    ASSERT_TRUE(device_tensor != nullptr);

    auto round = device_to_host<double>(device_tensor);
    ASSERT_EQ(round.numel(), 27);
    ASSERT_TRUE(round.eq(7.5).all().item<bool>());
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(HostToDeviceTests, EmptyTensor) {
    // shape [0], dtype float
    torch::Tensor t = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32));
    auto device_tensor = host_to_device<float>(t);
    ASSERT_TRUE(device_tensor != nullptr);

    auto round = device_to_host<float>(device_tensor);
    ASSERT_EQ(round.numel(), 0);
    ASSERT_TRUE(round.eq(0).all().item<bool>());  // empty tensor allclose zero trivially
}

TEST(HostToDeviceTests, NonContiguousTensor) {
    // Create a [4,4] then transpose to make strides non-standard
    torch::Tensor t = torch::arange(0, 16, torch::TensorOptions().dtype(torch::kInt32))
                          .view({4,4})
                          .transpose(0,1);
    ASSERT_FALSE(t.is_contiguous());
    auto device_tensor = host_to_device<int32_t>(t);
    ASSERT_TRUE(device_tensor != nullptr);

    auto round = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(round.sizes(), t.sizes());
    ASSERT_EQ(round.strides(), t.strides());
    ASSERT_TRUE(round.allclose(t));
}

TEST(HostToDeviceTests, HigherRankTensor) {
    // 5D tensor
    std::vector<int64_t> dims{2,2,2,2,2};
    torch::Tensor t = torch::rand(dims, torch::TensorOptions().dtype(torch::kFloat64));
    auto device_tensor = host_to_device<double>(t);
    ASSERT_TRUE(device_tensor != nullptr);

    auto round = device_to_host<double>(device_tensor);
    ASSERT_EQ(round.sizes(), t.sizes());
    ASSERT_TRUE(round.allclose(t));
}

/************************************************************************************************
 * Error conditions for host_to_device
 ***********************************************************************************************/

 VALIDATION_TEST(HostToDeviceTests, ThrowsOnTypeMismatch) {
    // create int tensor but call with wrong template arg
    torch::Tensor t = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));
    EXPECT_THROW(host_to_device<float>(t), std::runtime_error);
}

VALIDATION_TEST(HostToDeviceTests, ThrowsOnNullDataPtr) {
    // manually create a tensor with no storage (unusual, but data_ptr() == nullptr)
    torch::Tensor t = torch::Tensor();
    ASSERT_EQ(t.defined(), false);
    EXPECT_THROW(host_to_device<int32_t>(t), std::runtime_error);
}


/****************************************************************************************
 ****************************************************************************************
 ****                                                                                ****
 ****                            DEVICE_TO_HOST TESTS                                ****
 ****                                                                                ****
 ****************************************************************************************
 ****************************************************************************************/

/************************************************************************************************
 * Basic functionality
 ***********************************************************************************************/

 TEST(DeviceToHostTests, Basic1DInt64) {
    std::vector<int64_t> dims{4};
    auto device_tensor = zeros<int64_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);

    torch::Tensor t = device_to_host<int64_t>(device_tensor);
    ASSERT_EQ(t.sizes(), torch::IntArrayRef(dims));
    ASSERT_EQ(t.numel(), 4);
    // all elements should be zero
    ASSERT_TRUE(t.eq(0).all().item<bool>());
}

TEST(DeviceToHostTests, Basic2DInt32) {
    std::vector<int64_t> dims{2, 3};
    auto device_tensor = zeros<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);

    torch::Tensor t = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(t.numel(), 6);
    ASSERT_TRUE(t.eq(0).all().item<bool>());
}

TEST(DeviceToHostTests, Basic3DFloat) {
    std::vector<int64_t> dims{3, 3, 3};
    auto device_tensor = zeros<float>(dims);
    ASSERT_TRUE(device_tensor != nullptr);

    torch::Tensor t = device_to_host<float>(device_tensor);
    ASSERT_EQ(t.numel(), 27);
    ASSERT_TRUE(t.eq(0.0f).all().item<bool>());
}

/************************************************************************************************
 * Round‐trip consistency (via host_to_device)
 ***********************************************************************************************/

TEST(DeviceToHostTests, RoundTripDouble) {
    // start with a host tensor of random doubles
    torch::Tensor orig = torch::rand({2,4}, torch::TensorOptions().dtype(torch::kFloat64));
    auto device_tensor = host_to_device<double>(orig);
    ASSERT_TRUE(device_tensor != nullptr);

    // bring it back
    torch::Tensor recovered = device_to_host<double>(device_tensor);
    ASSERT_EQ(recovered.sizes(), orig.sizes());
    ASSERT_TRUE(recovered.allclose(orig));
}

/************************************************************************************************
 * Edge & corner cases
 ***********************************************************************************************/

TEST(DeviceToHostTests, EmptyTensor) {
    std::vector<int64_t> dims{0};
    auto device_tensor = zeros<int64_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);

    torch::Tensor t = device_to_host<int64_t>(device_tensor);
    ASSERT_EQ(t.numel(), 0);
    ASSERT_TRUE(t.eq(0).all().item<bool>());  // empty tensor is trivially “all zero”
}

TEST(DeviceToHostTests, ZeroSizeDimension) {
    std::vector<int64_t> dims{0, 5};
    auto device_tensor = zeros<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);

    torch::Tensor t = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(t.numel(), 0);
    ASSERT_TRUE(t.eq(0).all().item<bool>());
}

TEST(DeviceToHostTests, LargeTensor) {
    std::vector<int64_t> dims{500, 500};
    auto device_tensor = zeros<int32_t>(dims);
    ASSERT_TRUE(device_tensor != nullptr);

    torch::Tensor t = device_to_host<int32_t>(device_tensor);
    ASSERT_EQ(t.numel(), 500 * 500);
    ASSERT_TRUE(t.eq(0).all().item<bool>());
}

TEST(DeviceToHostTests, PreserveNonContiguousStrides) {
    // create a non-contiguous host tensor, round-trip to device
    torch::Tensor orig = torch::arange(0, 16, torch::TensorOptions().dtype(torch::kInt32))
                             .view({4,4})
                             .transpose(0,1);
    ASSERT_FALSE(orig.is_contiguous());
    auto device_tensor = host_to_device<int32_t>(orig);
    ASSERT_TRUE(device_tensor != nullptr);

    torch::Tensor t = device_to_host<int32_t>(device_tensor);
    // should preserve both shape and strides
    ASSERT_EQ(t.sizes(), orig.sizes());
    ASSERT_EQ(t.strides(), orig.strides());
    ASSERT_TRUE(t.allclose(orig));
}
