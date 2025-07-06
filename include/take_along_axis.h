#ifndef PERMUTE_H
#define PERMUTE_H


namespace lattica_hw_api {

/**
 * @brief Gathers elements from a tensor along a specified axis using provided indices.
 *
 * This function selects values from the input tensor `a` according to `indices`
 * along the specified `axis` and writes the result to the output tensor `result`.
 *
 * Expected Input/Output Shapes:
 * - Input tensor `a`: Arbitrary shape `[d0, ..., dn]`
 * - Indices tensor `indices`: Shape matches `a`, except at dimension `axis`, where
 *   it specifies which elements to take: `a.shape[:axis] + [k] + a.shape[axis+1:]`
 * - Output tensor `result`: Shape matches `indices`
 *
 * Arguments:
 * - `a` (input): Tensor to gather values from.
 * - `indices` (input): Tensor of indices indicating which values to select along the specified axis.
 * - `axis` (input): Axis along which to select values.
 * - `result` (output): Tensor to write the gathered values to.
 *
 * Notes:
 * - Each value in `indices` must be in the valid range for the dimension of `a` along `axis`.
 * - `result` must be preallocated with the correct shape (matching `indices`).
 * - All tensors must reside on compatible devices.
 */

template <typename T>
void take_along_axis(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<int64_t>>& indices,
    int64_t axis,
    std::shared_ptr<DeviceTensor<T>>& result
);

}

#endif // PERMUTE_H
