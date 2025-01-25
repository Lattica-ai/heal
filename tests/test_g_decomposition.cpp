#include "gtest/gtest.h"
#include "lattica_hw_api.h"

TEST(GDecompositionTests, Decompose) {
    NestedVectorType<int32_t, 1> aa_cpu = {51, 29, 63}; // [3]
    size_t power = 6;
    size_t base_bits = 1;

    auto aa_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(aa_cpu);
    auto decomposition_hw = lattica_hw_api::allocate_on_hardware<int32_t>({aa_cpu.size(), power});

    lattica_hw_api::g_decomposition(aa_hw, decomposition_hw, power, base_bits);
    NestedVectorType<int32_t, 2> decomposition_cpu = lattica_hw_api::move_from_hardware<int32_t, 2>(decomposition_hw);

    // Expected values (computed manually)
    NestedVectorType<int32_t, 2> expected_cpu = {{1, 1, 0, 0, 1, 1}, {1, 0, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1}};
    ASSERT_EQ(decomposition_cpu, expected_cpu);
}
