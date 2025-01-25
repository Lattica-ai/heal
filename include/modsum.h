#ifndef MODSUM_H
#define MODSUM_H

/**
 * @file modsum.h
 * @brief Provides elementwise modular summation operations over vectors.
 *
 * This module implements functions to perform modular arithmetic
 * (summation modulo) on vectors stored in device memory. These operations
 * are designed for hardware-accelerated platforms like GPUs or specialized hardware
 * and operate directly on pre-allocated device memory.
 *
 * Expected Input Sizes:
 * - Vector `a` must have dimensions `[m, k]`.
 * - Vector `b` can be a vector of dimensions `[m, k]`, `[k]` or a scalar.
 * - Moduli vector `p` must be a vector of size `[k]`.
 * - Output vector `result` must be pre-allocated with dimensions `[m, k]`.
 */

namespace lattica_hw_api {

    template <typename T>
    void modsum_v1(
        const std::shared_ptr<DeviceMemory<T>>& a,  // [m, k]
        const std::shared_ptr<DeviceMemory<T>>& b,  // [m, k]
        const std::shared_ptr<DeviceMemory<T>>& p,  //    [k]
        std::shared_ptr<DeviceMemory<T>>& result    // [m, k] (output)
    );

    template <typename T>
    void modsum_v2(
        const std::shared_ptr<DeviceMemory<T>>& a,  // [m, k]
        const std::shared_ptr<DeviceMemory<T>>& b,  //    [k]
        const std::shared_ptr<DeviceMemory<T>>& p,  //    [k]
        std::shared_ptr<DeviceMemory<T>>& result    // [m, k] (output)
    );

    template <typename T>
    void modsum_v3(
        const std::shared_ptr<DeviceMemory<T>>& a,  // [m, k]
        T b,                              //  scalar
        const std::shared_ptr<DeviceMemory<T>>& p,  //    [k]
        std::shared_ptr<DeviceMemory<T>>& result    // [m, k] (output)
    );

}

#endif // MODSUM_H
