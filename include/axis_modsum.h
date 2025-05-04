#ifndef AXIS_MODSUM_H
#define AXIS_MODSUM_H

namespace lattica_hw_api {

    /**
     * @brief Provides axis-wise modular summation operations over tensors.
     *
     * This module implements functions to perform modular summation along a specified axis
     * on tensors stored in device memory. Each summation is performed modulo a per-column
     * modulus vector `p`, which should match the size of the last dimension.
     *
     * Requirements:
     * - Tensor `a` must have shape `[..., k]`, where `k = p->dims[0]`.
     * - Tensor `p` must be a 1D tensor of shape `[k]`.
     * - Tensor `result` must have the same shape as `a` with the `axis` dimension removed.
     * - The reduction is performed along the given `axis`, and results are reduced modulo `p`.
     *
     * Example:
     * - If `a` has shape [m, s, k] and `axis = 1`, `result` must have shape [m, k].
     */


    template <typename T>
    void axis_modsum(
        const std::shared_ptr<DeviceTensor<T>>& a,        // input tensor [..., k]
        const std::shared_ptr<DeviceTensor<T>>& p,        // modulus [k]
        std::shared_ptr<DeviceTensor<T>>& result,         // output tensor [..., k] with axis removed
        int64_t axis                                      // axis to reduce
    );


    /**
     * @brief Element-wise modular multiply of a and b (with optional permute of b), then sum over the sum axis.
     *
     * @tparam T         Scalar type.
     * @param a          Tensor of shape [..., sum_size, k].
     * @param b          Tensor of shape [..., sum_size, k], broadcastable to a.
     * @param p          1D modulus tensor [k], all entries > 0.
     * @param perm       1D permutation of length matching the axis before the sum axis.
     * @param result     Output tensor of shape [..., k].
     * @param apply_perm If true, apply perm to b before multiplying.
     */
    template <typename T>
    void modmul_axis_sum(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<T>>& b,
        const std::shared_ptr<DeviceTensor<T>>& p,
        const std::shared_ptr<DeviceTensor<int64_t>>& perm,
        std::shared_ptr<DeviceTensor<T>>& result,
        bool apply_perm
    );
}

#endif // AXIS_MODSUM_H
