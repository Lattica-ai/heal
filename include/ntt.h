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
 * - Input tensor `a` must have shape `[l, m, r, k]` where:
 *     - `l` is the left batch dimension.
 *     - `m` is the transform length (must be a power of 2).
 *     - `r` is the right batch dimension.
 *     - `k` is the number of independent moduli.
 * - Modulus tensor `p` must have shape `[k]`.
 * - Permutation tensor `perm` must have shape `[m]`.
 * - Twiddle factors `twiddles` must have shape `[k, m]`.
 * - Modular inverses of `m`, `m_inv`, must have shape `[k]`.
 * - Output tensor `result` must have shape `[l, m, r, k]`.
 */

namespace lattica_hw_api {

    template <typename T>
    void ntt(
        const std::shared_ptr<DeviceTensor<T>>& a,        // [l, m, r, k]
        const std::shared_ptr<DeviceTensor<T>>& p,        // [k]
        const std::shared_ptr<DeviceTensor<T>>& perm,     // [m]
        const std::shared_ptr<DeviceTensor<T>>& twiddles, // [k, m]
        std::shared_ptr<DeviceTensor<T>>& result          // [l, m, r, k] (output)
    );

    template <typename T>
    void intt(
        const std::shared_ptr<DeviceTensor<T>>& a,             // [l, m, r, k]
        const std::shared_ptr<DeviceTensor<T>>& p,             // [k]
        const std::shared_ptr<DeviceTensor<T>>& perm,          // [m]
        const std::shared_ptr<DeviceTensor<T>>& inv_twiddles,  // [k, m]
        const std::shared_ptr<DeviceTensor<T>>& m_inv,         // [k]
        std::shared_ptr<DeviceTensor<T>>& result               // [l, m, r, k] (output)
    );

}

#endif // NTT_H
