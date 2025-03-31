#include "gtest/gtest.h"
#include "lattica_hw_api.h"

using namespace lattica_hw_api;

TEST(NonContiguousTests, TransposeAndAddWithStrides) {
    // Create a [2, 3, 3] tensor
    torch::Tensor a = torch::arange(2 * 3 * 3, torch::kInt32).reshape({2, 3, 3, 1});
    torch::Tensor a_t = a.transpose(1, 2);  // [2, 3, 3, 1], non-contiguous

    ASSERT_EQ(a.sizes(), a_t.sizes());
    ASSERT_FALSE(a_t.is_contiguous());

    // Move both tensors to device
    auto a_hw = host_to_device<int32_t>(a);
    auto a_t_hw = host_to_device<int32_t>(a_t);

    // Allocate result and modulus
    auto result_hw = allocate_on_hardware<int32_t>({2, 3, 3, 1});
    auto p_hw = host_to_device<int32_t>(torch::tensor({100}, torch::kInt32));  // [1]

    // Perform modular multiplication: result = a * a_t mod 100
    modmul_ttt(a_hw, a_t_hw, p_hw, result_hw);

    result_hw->print();

    // Verify
    torch::Tensor expected = (a * a_t).remainder(100);
    torch::Tensor result = device_to_host<int32_t>(result_hw);

    ASSERT_TRUE(torch::equal(result, expected)) << "modmul_ttt failed on transpose with noncontiguous strides.";
}
