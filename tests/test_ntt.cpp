#include "gtest/gtest.h"
#include "lattica_hw_api.h"

TEST(NTTTests, PerformNTTAndVerifyRestoration) {
    NestedVectorType<int32_t, 3> a_cpu = {{{1, 2}, {3, 4}, {5, 6}, {7, 8}}}; // [1, 4, 2]
    NestedVectorType<int32_t, 1> p_cpu = {17, 257};                                 // [2]
    NestedVectorType<int32_t, 1> m_inv_cpu = {13, 193};                            // [2]
    NestedVectorType<int32_t, 1> perm_cpu = {0, 2, 1, 3};                          // [4]
    NestedVectorType<int32_t, 2> twiddles_cpu = {{1, 1}, {4, 16}, {2, 4}, {8, 64}}; // [4, 2]
    NestedVectorType<int32_t, 2> inv_twiddles_cpu = {{1, 1}, {13, 241}, {9, 193}, {15, 253}}; // [4, 2]

    auto a_hw = lattica_hw_api::move_to_hardware<int32_t, 3>(a_cpu);
    auto p_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(p_cpu);
    auto m_inv_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(m_inv_cpu);
    auto perm_hw = lattica_hw_api::move_to_hardware<int32_t, 1>(perm_cpu);
    auto twiddles_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(twiddles_cpu);
    auto inv_twiddles_hw = lattica_hw_api::move_to_hardware<int32_t, 2>(inv_twiddles_cpu);

    auto result_hw = lattica_hw_api::allocate_on_hardware<int32_t>({1, 4, 2});
    auto restored_hw = lattica_hw_api::allocate_on_hardware<int32_t>({1, 4, 2});

    lattica_hw_api::ntt<int32_t>(a_hw, p_hw, perm_hw, twiddles_hw, result_hw);
    lattica_hw_api::intt<int32_t>(result_hw, p_hw, perm_hw, inv_twiddles_hw, m_inv_hw, restored_hw);

    NestedVectorType<int32_t, 3> restored_cpu = lattica_hw_api::move_from_hardware<int32_t, 3>(restored_hw);

    ASSERT_EQ(a_cpu, restored_cpu) << "Restored input does not match the original input.";
}
