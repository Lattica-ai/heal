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
 * - Input vector `a` must have dimensions `[l, m, k]`.
 * - Input vector `p` must have dimensions `[k]`.
 * - Permutation array `perm` must have dimensions `[m]`.
 * - Twiddle factors `twiddles` must have dimensions `[m, k]`.
 * - Modular inverses of m `m_inv` must have dimensions `[k]`.
 * - Result vector `result` must have dimensions `[l, m, k]` (output).
 */

namespace lattica_hw_api {

    template <typename T>
    void ntt(
        const std::shared_ptr<DeviceMemory<T>>& a,        // [l, m, k]
        const std::shared_ptr<DeviceMemory<T>>& p,        // [k]
        const std::shared_ptr<DeviceMemory<T>>& perm,     // [m]
        const std::shared_ptr<DeviceMemory<T>>& twiddles, // [m, k]
        std::shared_ptr<DeviceMemory<T>>& result          // [l, m, k] (output)
    );

    template <typename T>
    void intt(
        const std::shared_ptr<DeviceMemory<T>>& a,             // [l, m, k]
        const std::shared_ptr<DeviceMemory<T>>& p,             // [k]
        const std::shared_ptr<DeviceMemory<T>>& perm,          // [m]
        const std::shared_ptr<DeviceMemory<T>>& inv_twiddles,  // [m, k]
        const std::shared_ptr<DeviceMemory<T>>& m_inv,         // [k]
        std::shared_ptr<DeviceMemory<T>>& result               // [l, m, k] (output)
    );

}

#endif // NTT_H
