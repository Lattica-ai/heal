#ifndef NTT_H
#define NTT_H

/**
 * @file ntt.h
 * @brief Provides the implementation of the Number Theoretic Transform (NTT).
 *
 * This module implements the NTT, a specialized discrete Fourier transform
 * used in modular arithmetic. It transforms data for efficient polynomial
 * multiplication and other applications.
 *
 * Expected Input Sizes:
 * - Input tensor `a` must have shape `[l, m, r, k]` or `[l, r, k, m]`, where:
 *     - `l` is the left batch dimension.
 *     - `m` is the transform length (must be a power of 2).
 *     - `r` is the right batch dimension.
 *     - `k` is the number of independent moduli.
 * - Modulus tensor `p` must have shape `[k]`.
 * - Permutation tensor `perm` must have shape `[m]`.
 * - Twiddle factors `twiddles` must have shape `[k, m]`.
 * - Modular inverses of `m`, `m_inv`, must have shape `[k]`.
 * - Axis of `m` the transform length, `axis` can be -3 (for `[l, m, r, k]`) or -1 (for `[l, r, k, m]`).
 * - Output tensor `result` must have shape `[l, m, r, k]` or `[l, r, k, m]`.
 *
 * Optional Barrett Reduction Parameters:
 * - `log2p_list` (shape `[k]`) – precomputed ⌊log₂(pᵢ)⌋ for each modulus pᵢ.
 * - `mu_list`    (shape `[k]`) – precomputed Barrett constant ⌊2²ⁿ / pᵢ⌋ for each modulus pᵢ.
 */

namespace lattica_hw_api {

    template <typename T>
    void ntt(
        const std::shared_ptr<DeviceTensor<T>>& a,          // [l, m, r, k] or [l, r, k, m]
        const std::shared_ptr<DeviceTensor<T>>& p,          // [k]
        const std::shared_ptr<DeviceTensor<T>>& perm,       // [m]
        const std::shared_ptr<DeviceTensor<T>>& twiddles,   // [k, m]
        const std::shared_ptr<DeviceTensor<T>>& log2p_list, // [k]
        const std::shared_ptr<DeviceTensor<T>>& mu_list,    // [k]
        int64_t axis,                                       // Axis of m
        std::shared_ptr<DeviceTensor<T>>& result            // [l, m, r, k] or [l, r, k, m] (output)
    );

    template <typename T>
    void intt(
        const std::shared_ptr<DeviceTensor<T>>& a,             // [l, m, r, k]
        const std::shared_ptr<DeviceTensor<T>>& p,             // [k]
        const std::shared_ptr<DeviceTensor<T>>& perm,          // [m]
        const std::shared_ptr<DeviceTensor<T>>& inv_twiddles,  // [k, m]
        const std::shared_ptr<DeviceTensor<T>>& m_inv,         // [k]
        const std::shared_ptr<DeviceTensor<T>>& log2p_list,    // [k]
        const std::shared_ptr<DeviceTensor<T>>& mu_list,       // [k]
        std::shared_ptr<DeviceTensor<T>>& result               // [l, m, r, k] (output)
    );

}

#endif // NTT_H
