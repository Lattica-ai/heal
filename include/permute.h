#ifndef PERMUTE_H
#define PERMUTE_H

/**
 * @file permute.h
 * @brief Applies elementwise permutations along two axes of a multi-dimensional tensor.
 *
 * This module applies permutations to slices of a tensor along two specified axes.
 * The permutation is performed independently for each pair of indices along the remaining axes
 * (batch dimensions).
 *
 * Inputs:
 * - Tensor `a` of arbitrary shape.
 * - Axis indices `perm_axis` and `elementwise_axis`:
 *     - `a.shape[perm_axis] = m`
 *     - `a.shape[elementwise_axis] = l`
 * - Permutation tensor `perms` of shape `[l, m]` specifying the new ordering of `m` elements
 *   for each of the `l` elements along `elementwise_axis`.
 *
 * Output:
 * - Tensor `result` with the **same shape as `a`**, where each `[l, m, ...]` slice is permuted accordingly.
 *
 * Requirements:
 * - `0 <= perm_axis < a.ndim`
 * - `0 <= elementwise_axis < a.ndim`
 * - `perm_axis != elementwise_axis`
 * - `a.shape[elementwise_axis] == perms.shape[0] == l`
 * - `a.shape[perm_axis] == perms.shape[1] == m`
 */

namespace lattica_hw_api {

    template <typename T>
    void permute(
        const std::shared_ptr<DeviceTensor<T>>& a,          // [..., l, ..., m, ...]
        const std::shared_ptr<DeviceTensor<T>>& perms,      // [l, m]
        std::shared_ptr<DeviceTensor<T>>& result,           // same shape as `a`
        int64_t elementwise_axis,                           // axis with l elements (used as rows)
        int64_t perm_axis                                   // axis with m elements (to permute)
    );

}

#endif // PERMUTE_H
