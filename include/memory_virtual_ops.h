#ifndef MEMORY_OPS_H
#define MEMORY_OPS_H

#include <variant>
/**
 * @file memory_ops.h
 * @brief Provides virtual and utility memory operations for DeviceTensor tensors.
 *
 * This module defines a collection of lightweight, zero-copy transformations and
 * memory-related operations that can be applied to DeviceTensor tensors.
 * All operations modify tensor metadata (dims and strides) in-place to simulate
 * broadcast-style behavior without duplicating or reallocating memory.
 *
 * Currently supported operations:
 * - expand: Repeats elements along a given axis using stride manipulation.
 * - squeeze: Removes a size-1 dimension at the specified axis.
 * - unsqueeze: Inserts a new size-1 dimension at the specified axis.
 */

namespace lattica_hw_api {

    /**
     * @brief Expands a tensor in-place by virtually repeating elements along the specified axis.
     *        This is done by modifying the dimension and setting the stride to zero.
     *        Only scalar `repeats` are supported.
     *
     * Example:
     * Given a tensor of shape [2, 1] and repeat = 3 along axis = 1,
     * the new shape becomes [2, 3], with repeated elements along axis 1.
     *
     * Preconditions:
     * - The specified axis must be valid and must have size 1 in the input.
     * - The repeat count must be positive.
     *
     * @tparam T The element type.
     * @param tensor The input tensor to be expanded. Its dims and strides will be modified.
     * @param axis The axis along which to repeat.
     * @param repeats The number of times to repeat elements (must be > 0).
     */
    template <typename T>
    std::shared_ptr<DeviceTensor<T>> expand(
        const std::shared_ptr<DeviceTensor<T>>& tensor,
        int64_t axis,
        int64_t repeats
    );

    /**
     * @brief Removes a size-1 dimension at the specified axis.
     *        Modifies the dims and strides in-place.
     *
     * Example:
     * Given a tensor of shape [3, 1, 4], squeeze at axis = 1 → [3, 4]
     *
     * Preconditions:
     * - The specified axis must be valid and must be of size 1.
     *
     * @tparam T The element type.
     * @param tensor The input tensor to squeeze. Modified in-place.
     * @param axis The axis to remove.
     * @return A pointer to the modified tensor.
     */
    template <typename T>
    std::shared_ptr<DeviceTensor<T>> squeeze(
        const std::shared_ptr<DeviceTensor<T>>& tensor,
        int64_t axis
    );

    /**
     * @brief Inserts a new dimension of size 1 at the specified axis.
     *        Modifies the dims and strides in-place.
     *
     * Example:
     * Given a tensor of shape [3, 4], unsqueeze at axis = 1 → [3, 1, 4]
     *
     * Preconditions:
     * - The axis must be in the range [-ndim-1, ndim]
     *
     * @tparam T The element type.
     * @param tensor The input tensor to unsqueeze. Modified in-place.
     * @param axis The position to insert the new axis.
     * @return A pointer to the modified tensor.
     */
    template <typename T>
    std::shared_ptr<DeviceTensor<T>> unsqueeze(
        const std::shared_ptr<DeviceTensor<T>>& tensor,
        int64_t axis
    );



    struct Slice {
        int64_t start;   // inclusive
        int64_t end;     // exclusive
        int64_t step;    // >0
        //
        // Convenience constructor:
        Slice(int64_t _start, int64_t _end, int64_t _step = 1)
          : start(_start), end(_end), step(_step) {}
    };

    /**
     * @brief Specifies either a single‐element index or a slice over an axis.
     *
     * This variant drives the behavior of get_slice():
     *  - `int64_t` means “take exactly this index on the axis, collapsing that dimension.”
     *  - `Slice`   means “take elements from start (inclusive) to end (exclusive),
     *                 stepping by `step` each time.”
     *
     * @see Slice, get_slice()
     */
    using SliceArg = std::variant<int64_t /*a single index*/, Slice>;


    /**
     * @brief Returns a zero‐copy view of the input tensor, sliced along each axis.
     *
     * In‐place modifies the tensor’s dims, strides, and data pointer (via
     * a shared_ptr alias) so that no new buffer is allocated. Collapsed
     * dimensions (where you passed an `int64_t` index) are removed from
     * the output shape.
     *
     * Example:
     *   // From a [2×3×4] tensor, pick block 1, then rows [0,2), cols [1,4):
     *   auto view = get_slice<float>(orig,
     *       { int64_t(1),
     *         Slice(0,2),     // rows 0 and 1
     *         Slice(1,4) });  // cols 1,2,3
     *
     * Preconditions:
     *  - `slices.size() == input->dims.size()`
     *  - For each axis:
     *      * If `int64_t idx`,     0 ≤ idx < dim_size
     *      * If `Slice(s,e,st)`,   0 ≤ s < e ≤ dim_size and st > 0
     *
     * @tparam T Element type stored in the tensor.
     * @param input   The tensor to slice; its metadata is updated in‐place.
     * @param slices  One SliceArg per axis of `input`.
     * @return A pointer to the modified tensor.
     */
    template<typename T>
    std::shared_ptr<DeviceTensor<T>> get_slice(
        const std::shared_ptr<DeviceTensor<T>>& input,
        const std::vector<SliceArg>& slices
);

}

#endif // MEMORY_OPS_H
