#pragma once

#include "device_tensor.h"
#include <memory>

namespace lattica_hw_api {

/**
 * @brief Provides decomposition of tensor elements into digits in a specified base.
 *
 * This module computes the decomposition of elements from an input tensor
 * into multiple digits in a given base and stores the results in an output tensor.
 *
 * Expected Input/Output Shapes:
 * - Input tensor `a` can have arbitrary shape: `[...,]`
 * - `power` specifies the number of digits to compute.
 * - `base_bits` specifies the number of bits in the base (i.e., base = 2^base_bits).
 * - Output tensor `result` must have shape: `a.shape + [power]`
 *
 * Notes:
 * - Each input element is decomposed into `power` base-2^base_bits digits.
 * - Results are stored along a new final axis of size `power`.
 */

 template <typename T, typename U>
 void apply_g_decomp(
     const std::shared_ptr<DeviceTensor<T>>& a,         // [...], arbitrary shape
     size_t power,                                      // Number of digits
     size_t base_bits,                                  // Base bits
     std::shared_ptr<DeviceTensor<U>>& result           // [..., power] (output)
 );

} // namespace lattica_hw_api