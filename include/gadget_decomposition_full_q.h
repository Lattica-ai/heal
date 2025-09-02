#pragma once

#include "device_tensor.h"
#include <memory>

namespace lattica_hw_api {

/**
 * @brief Performs gadget decomposition on RNS-represented tensors into base-g digit representation.
 *
 * This function converts tensor elements from Chinese Remainder Theorem (CRT/RNS) representation
 * into gadget decomposition form, where each element is decomposed into g_exp digits in base 2^g_base_bits.
 * The decomposition is performed relative to the full modulus product Q = ∏q_i.
 *
 * Expected Input/Output Shapes:
 * - Input tensor `a`: `[reps_l, q_list_len, reps_r]` - RNS representation
 * - q_list: `[q_list_len]` - RNS moduli
 * - g_exp: Number of digits to extract
 * - g_base_bits: Number of bits per digit
 * - Output tensor `out`: `[reps_l, g_exp, reps_r]` - Gadget decomposition digits
 */
template <typename T, typename U>
void apply_g_decomp_relative_to_full_q(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& q_list,
    int g_exp,
    int g_base_bits,
    std::shared_ptr<DeviceTensor<U>>& out
);

} // namespace lattica_hw_api