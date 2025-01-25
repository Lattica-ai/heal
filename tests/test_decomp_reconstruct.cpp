#include "gtest/gtest.h"
#include "lattica_hw_api.h"

TEST(DecompReconstructTests, DecomposeAndReconstruct) {
    NestedVectorType<int32_t, 1> a_cpu = {51, 29, 63}; // [3]
    size_t power = 6;
    size_t base_bits = 1;

    auto a_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(a_cpu);
    auto decomposition_hw = lattica_hw_api::allocate_on_hardware<int32_t>({a_cpu.size(), power});

    lattica_hw_api::g_decomposition(a_hw, decomposition_hw, power, base_bits);

    NestedVectorType<int32_t, 1> basis_cpu = {1, 2, 4, 8, 16, 32}; // [power]
    auto basis_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(basis_cpu);

    // Allocate memory for multiplication result
    auto multiplication_result_hw = lattica_hw_api::allocate_on_hardware<int32_t>({a_cpu.size(), power});

    auto p_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(NestedVectorType<int32_t, 1>(6, 1024));

    // Perform modular multiplication
    lattica_hw_api::modmul_v2(decomposition_hw, basis_hw, p_hw, multiplication_result_hw);

    // Reshape with a redundant dimension to obay the axis_modsum API
    multiplication_result_hw->reshape({a_cpu.size(), power, 1});

    auto reconstruction_hw = lattica_hw_api::allocate_on_hardware<int32_t>({a_cpu.size(), 1});
    p_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(NestedVectorType<int32_t, 1>(1, 1024)); // Modulo
    lattica_hw_api::axis_modsum(multiplication_result_hw, p_hw, reconstruction_hw);

    NestedVectorType<int32_t, 1> reconstruction_cpu = lattica_hw_api::move_from_hardware<int32_t, 1>(reconstruction_hw);
    ASSERT_EQ(reconstruction_cpu, a_cpu) << "Reconstruction failed.";
}
