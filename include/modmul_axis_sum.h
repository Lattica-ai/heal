#pragma once

#include "device_tensor_ex_impl.h"

namespace lattica_hw_api {

/**
 * @brief Computes a modular axis-wise sum-product between tensors `a` and `b`, **accumulating** into the output tensor.
 *
 * This function performs a modular sum over a specified axis, computing:
 *     result[...] += sum_i (a[...] * b[...]) % p
 * The contraction axis and output shape depend on the specified `axis` parameter.
 *
 * ## Accumulation Behavior
 * **IMPORTANT:**
 * - The `result` tensor is both an input and output.
 * - **This function *adds* its computed values to the existing values in `result`.**
 * - If you want a "fresh" output, you must zero-initialize `result` before the first call.
 * - If you call this function multiple times, each call will *accumulate* results onto the previous values.
 *
 * ## Supported Input Shapes:
 * - For axis = -1:
 *     - a:      [reps, sum_size, k, n]
 *     - b:      [sum_size, k, n]
 *     - result: [reps, k, n]
 * - For axis = -3:
 *     - a:      [reps, n, sum_size, k]
 *     - b:      [n, sum_size, k]
 *     - result: [reps, n, k]
 * - Modulus tensor `p` must have shape [k].
 * - Permutation tensor `perm` must have shape [n], only used if `apply_perm` is true.
 * - Optional parameters `log2p_list` and `mu_list` are for Barrett reduction (not required for correctness).
 *
 * ## Arguments:
 * @tparam T         Element type (e.g., int64_t)
 * @param a          Input tensor `a` (see shape above)
 * @param b          Input tensor `b` (see shape above)
 * @param p          Modulus tensor, shape [k]
 * @param perm       Permutation tensor, shape [n] (used only if `apply_perm` is true)
 * @param log2p_list Optional: Barrett log2(p) constants, shape [k]
 * @param mu_list    Optional: Barrett mu constants, shape [k]
 * @param axis       Contraction axis. Must be -1 or -3.
 * @param apply_perm Whether to apply permutation from `perm` to the contraction axis
 * @param result     Output tensor (see shape above)
 *
 * ## Behavior:
 * - For axis == -1: Sums over axis 1 of `a` and axis 0 of `b`.
 * - For axis == -3: Sums over axis 2 of `a` and axis 1 of `b`.
 * - If `apply_perm` is true, the specified axis is permuted according to `perm`.
 * - Each multiplication is performed in modular arithmetic (using `p`), with overflow safety.
 * - **Result values are accumulated:** for every call, the function adds new results to the existing values in `result`.
 *
 * ## Throws:
 * - std::invalid_argument if input shapes or axis are invalid, or if moduli are not positive.
 *
 * ## Example usage:
 * ```
 * // a: [10, 3, 2, 4], b: [3, 2, 4], p: [2]
 * // Zero-initialize result if you do not want accumulation from previous calls
 * result->fill(0); // Pseudocode for initialization
 * modmul_axis_sum<int64_t>(a, b, p, nullptr, nullptr, nullptr, -1, false, result);
 *
 * // Call again to accumulate more into result:
 * modmul_axis_sum<int64_t>(a2, b2, p, nullptr, nullptr, nullptr, -1, false, result);
 * ```
 */

template <typename T>
void modmul_axis_sum(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& b,
    const std::shared_ptr<DeviceTensor<T>>& p,
    const std::shared_ptr<DeviceTensor<T>>& perm,
    const std::shared_ptr<DeviceTensor<T>>& log2p_list,
    const std::shared_ptr<DeviceTensor<T>>& mu_list,
    int64_t axis,
    bool apply_perm,
    std::shared_ptr<DeviceTensor<T>>& result
);

} // namespace lattica_hw_api
