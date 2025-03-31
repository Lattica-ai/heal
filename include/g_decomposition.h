#ifndef G_DECOMPOSITION_H
#define G_DECOMPOSITION_H

/**
 * @file g_decomposition.h
 * @brief Provides decomposition of tensor elements into digits in a specified base.
 *
 * This module computes the decomposition of elements from an input tensor
 * into multiple digits in a given base and stores the results in an output tensor.
 *
 * Expected Input/Output Shapes:
 * - Input tensor `a` can have arbitrary shape: `[...,]`
 * - Output tensor `result` must have shape: `a.shape + [power]`
 * - `power` specifies the number of digits to compute.
 * - `base_bits` specifies the number of bits in the base (i.e., base = 2^base_bits).
 *
 * Notes:
 * - Each input element is decomposed into `power` base-2^base_bits digits.
 * - Results are stored along a new final axis of size `power`.
 */

namespace lattica_hw_api {

    template <typename T>
    void g_decomposition(
        const std::shared_ptr<DeviceTensor<T>>& a,         // [...], arbitrary shape
        std::shared_ptr<DeviceTensor<T>>& result,          // [..., power] (output)
        size_t power,                                      // Number of digits
        size_t base_bits                                   // Base bits
    );

}

#endif // G_DECOMPOSITION_H
