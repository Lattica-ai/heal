#ifndef CONTIGUOUS_H
#define CONTIGUOUS_H

/**
 * @file contiguous.h
 * @brief Provides utilities for cenforcing contiguous layout for DeviceTensor.
 */

namespace lattica_hw_api {
    /**
     * @brief Returns a contiguous version of the input tensor.
     * If the tensor is already contiguous, returns the same tensor.
     * Otherwise, allocates a new buffer and copies data.
     *
     * @tparam T Element type.
     * @param tensor Input tensor.
     * @return A contiguous version of the tensor.
     */
    template <typename T>
    std::shared_ptr<DeviceTensor<T>> contiguous(const std::shared_ptr<DeviceTensor<T>>& tensor);

}

#endif // CONTIGUOUS_H
