#pragma once

#include "device_tensor.h"
#include <memory>

namespace lattica_hw_api {

/**
 * @file tensor_advanced_ops.h
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

template <typename T, typename U>
void apply_g_decomp(
    const std::shared_ptr<DeviceTensor<T>>& a,         // [...], arbitrary shape
    std::shared_ptr<DeviceTensor<U>>& result,          // [..., power] (output)
    size_t power,                                      // Number of digits
    size_t base_bits                                   // Base bits
);

/**
 * @brief Gathers elements from a tensor along a specified axis using provided indices.
 *
 * This function selects values from the input tensor `a` according to `indices`
 * along the specified `axis` and writes the result to the output tensor `result`.
 *
 * Expected Input/Output Shapes:
 * - Input tensor `a`: Arbitrary shape `[d0, ..., dn]`
 * - Indices tensor `indices`: Broadcast-compatible with `a` (same shape, or any dimension is 1), except at dimension `axis`, where
 *   it may be either 1 or match `a.shape[axis]`.
 *
 * Arguments:
 * - `a` (input): Tensor to gather values from.
 * - `indices` (input): Tensor of indices indicating which values to select along the specified axis.
 * - `axis` (input): Axis along which to select values.
 * - `result` (output): Tensor to write the gathered values to.
 *
 * Notes:
 * - Each value in `indices` must be in the valid range for the dimension of `a` along `axis`.
 * - `indices` must be broadcast-compatible with `a` (all dims either 1 or matching, and at the gather axis, 1 or matching).
 * - `result` must be preallocated with the correct shape.
 * - All tensors must reside on compatible devices.
 */

 template <typename T>
 void take_along_axis(
     const std::shared_ptr<DeviceTensor<T>>& a,
     const std::shared_ptr<DeviceTensor<int64_t>>& indices,
     int64_t axis,
     std::shared_ptr<DeviceTensor<T>>& result
 );

} // namespace lattica_hw_api