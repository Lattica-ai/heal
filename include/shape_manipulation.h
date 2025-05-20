#ifndef SHAPE_MANIPULATION_H
#define SHAPE_MANIPULATION_H

/**
 * @brief Append zeros to the end of one axis of a tensor.
 *
 * Given an input DeviceTensor `a`, this function writes into a pre-allocated
 * DeviceTensor `result` so that along the chosen axis:
 *
 *   result.shape[axis] == a.shape[axis] + pad
 *
 * All other dimensions are identical.  The extra entries are filled with zero.
 *
 * @tparam T   element type
 * @param a         shared_ptr to the source tensor (must be non-null)
 * @param pad       number of zeros to append (must be ≥ 0)
 * @param axis      axis along which to pad; may be negative (−1 is last axis)
 * @param result    shared_ptr to the destination tensor (must be non-null,
 *                  and already allocated with the correct expanded shape)
 *
 * @throws std::invalid_argument if
 *   - `a` or `result` is null
 *   - `pad < 0`
 *   - `axis` is out of range (−rank … rank−1)
 *   - `result->dims` does not match `a->dims` except for the specified axis
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
