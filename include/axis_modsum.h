#ifndef AXIS_MODSUM_H
#define AXIS_MODSUM_H

/**
 * @file axis_modsum.h
 * @brief Provides axis-wise modular summation operations over vectors.
 *
 * This module implements functions to perform modular arithmetic
 * (summation modulo) along specified axes on vectors stored in device memory.
 * These operations are designed for hardware-accelerated platforms like GPUs or specialized hardware
 * and operate directly on pre-allocated device memory.
 *
 * Expected Input Sizes:
 * - Tensor `a` must have dimensions `[m, s, k]`.
 * - Moduli vector `p` must be a vector of size `[k]`.
 * - Output vector `result` must be pre-allocated with dimensions `[m, k]`.
 */

namespace lattica_hw_api {

    template <typename T>
    void axis_modsum(
        const std::shared_ptr<DeviceMemory<T>>& a,  // [m, s, k]
        const std::shared_ptr<DeviceMemory<T>>& p,  //       [k]
        std::shared_ptr<DeviceMemory<T>>& result    //    [m, k] (output)
    );

}

#endif // AXIS_MODSUM_H
