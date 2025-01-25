#ifndef G_DECOMPOSITION_H
#define G_DECOMPOSITION_H

/**
 * @file g_decomposition.h
 * @brief Provides decomposition of vector elements into digits in a specified base.
 *
 * This module computes the decomposition of elements from an input vector
 * into multiple digits in a given base and stores the results in an output vector.
 *
 * Expected Input Sizes:
 * - Input vector `a` must have dimensions `[m]`.
 * - Result vector `result` must have dimensions `[m, power]`.
 * - `power` specifies the number of digits to compute.
 * - `base_bits` specifies the number of bits in the base (e.g., base = 2^base_bits).
 */

namespace lattica_hw_api {

    template <typename T>
    void g_decomposition(
        const std::shared_ptr<DeviceMemory<T>>& a,      // [m]
        std::shared_ptr<DeviceMemory<T>>& result,       // [m, power] (output)
        size_t power,                                // Number of digits
        size_t base_bits                             // Base bits
    );

}

#endif // G_DECOMPOSITION_H
