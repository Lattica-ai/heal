#include "gtest/gtest.h"
#include "lattica_hw_api.h"

using namespace lattica_hw_api;
using IntVec1D = NestedVectorType<int32_t, 1>;

// Define macros for move_to_hardware and allocate_on_hardware
#define TO_HW(vec) move_to_hardware<int32_t, 1>(vec)
#define ALLOC_HW(shape) allocate_on_hardware<int32_t>(shape)

TEST(DecompReconstructTests, DecomposeAndReconstruct) {
    IntVec1D a_cpu = {51, 29, 63}; // [3]
    size_t power = 6;
    size_t base_bits = 1;

    auto a_hw = TO_HW(a_cpu);
    auto a_digits_hw = ALLOC_HW((std::vector<size_t>{a_cpu.size(), power}));

    g_decomposition(a_hw, a_digits_hw, power, base_bits);

    auto basis_hw = TO_HW(IntVec1D({1, 2, 4, 8, 16, 32}));
    auto p6_hw = TO_HW(IntVec1D(6, 1024));

    // Perform modular multiplication inplace
    modmul_v2(a_digits_hw, basis_hw, p6_hw, a_digits_hw);

    // Reshape with a redundant dimension to obey the axis_modsum API
    a_digits_hw->reshape({a_cpu.size(), power, 1});

    auto a_recon_hw = ALLOC_HW((std::vector<size_t>{a_cpu.size(), 1}));
    auto p1_hw = TO_HW(IntVec1D(1, 1024));

    axis_modsum(a_digits_hw, p1_hw, a_recon_hw);

    IntVec1D reconstruction_cpu = move_from_hardware<int32_t, 1>(a_recon_hw);
    ASSERT_EQ(reconstruction_cpu, a_cpu) << "Reconstruction failed.";
}
