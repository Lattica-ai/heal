#pragma once

#include "device_tensor.h"
#include <memory>

namespace lattica_hw_api {

/**
 * @brief Sets all elements of the tensor to a constant value.
 *
 * This function assigns the given constant value `val` to every element
 * in the device tensor `a`.
 *
 * @tparam T  Data type of the tensor elements.
 * @param a   Shared pointer to the device tensor to modify.
 * @param val The constant value to set for all elements.
 */
template <typename T>
void set_const_val(
    const std::shared_ptr<DeviceTensor<T>>& a,
    T val
);


/**
 * @file pad_single_axis.h
 * @brief Appends zero values to the end of a specific axis in a tensor.
 *
 * This module pads a tensor by adding zero elements at the end of the specified axis,
 * returning a new tensor with an expanded shape along that axis.
 *
 * Inputs:
 * - Tensor `a` of arbitrary shape.
 * - Integer `pad`: number of zeros to append.
 * - Integer `axis`: axis along which to pad (may be negative to count from the end).
 *
 * Output:
 * - Tensor `result` with the same shape as `a` except that
 *   `result.shape[axis] = a.shape[axis] + pad`.
 *
 * Requirements:
 * - `pad >= 0`
 * - `axis` must satisfy `-a->ndim() <= axis < a->ndim()`
 *   (negative values count from the end: `-1` = last axis, `-2` = second-to-last, etc.).
 */
template <typename T>
void pad_single_axis(
    const std::shared_ptr<DeviceTensor<T>>& a,      // input tensor
    int64_t pad,                                    // number of zeros to append
    int64_t axis,                                   // axis index to pad
    std::shared_ptr<DeviceTensor<T>>& result        // output tensor
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

}; // namespace lattica_hw_api

