#ifndef SHAPE_MANIPULATION_H
#define SHAPE_MANIPULATION_H

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

namespace lattica_hw_api {

    template <typename T>
    void pad_single_axis(
        const std::shared_ptr<DeviceTensor<T>>& a,      // input tensor
        int64_t pad,                                    // number of zeros to append
        int64_t axis,                                   // axis index to pad
        std::shared_ptr<DeviceTensor<T>>& result        // output tensor
    );

}

#endif // SHAPE_MANIPULATION_H
