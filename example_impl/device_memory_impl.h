#ifndef DEVICEMEMORYIMPL_H
#define DEVICEMEMORYIMPL_H

#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <numeric>

#include "nested_vector.h"

/**
 * @file core.h
 * @brief Core components for managing hardware memory.
 *
 * DeviceMemory now supports vectors of up to 4 dimensions.
 * For vectors with fewer dimensions, higher dimensions are set to 1.
 */
template <typename T>
struct DeviceMemory {
    size_t dimensions;  // Active dimensions
    std::vector<size_t> dims;  // Vector of dimensions
    std::unique_ptr<T[]> data;  // Pointer to the memory data

    void print();

    /**
     * Constructor for up to 4D vectors.
     * For vectors with fewer dimensions, higher dimensions should be set to 1.
     * @param dims Vector of dimensions (size <= 4). Missing dimensions default to 1.
     */
    DeviceMemory(const std::vector<size_t>& dims)
        : dimensions(dims.size()),
          dims(dims),
          data(std::make_unique<T[]>(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()))) {}

    /**
     * Access an element in a vector by indices.
     * @param indices Vector of indices matching the vector dimensions.
     * @return Reference to the element.
     */
    T& at(const std::vector<size_t>& indices) {
        if (indices.size() != dimensions) {
            throw std::invalid_argument("Number of indices does not match vector dimensions.");
        }

        size_t flat_index = 0;
        size_t stride = 1;
        for (size_t i = dimensions; i-- > 0;) {
            if (indices[i] >= dims[i]) {
                throw std::out_of_range("Index out of range in DeviceMemory.");
            }
            flat_index += indices[i] * stride;
            stride *= dims[i];
        }
        return data[flat_index];
    }

    /**
     * Access an element in a vector by indices (const version).
     * @param indices Vector of indices matching the vector dimensions.
     * @return Const reference to the element.
     */
    const T& at(const std::vector<size_t>& indices) const {
        if (indices.size() != dimensions) {
            throw std::invalid_argument("Number of indices does not match vector dimensions.");
        }

        size_t flat_index = 0;
        size_t stride = 1;
        for (size_t i = dimensions; i-- > 0;) {
            if (indices[i] >= dims[i]) {
                throw std::out_of_range("Index out of range in DeviceMemory.");
            }
            flat_index += indices[i] * stride;
            stride *= dims[i];
        }
        return data[flat_index];
    }

    /**
     * Reshape the vector to new dimensions.
     * @param new_dims Vector of new dimensions (size <= 4).
     * Throws std::invalid_argument if the total size does not match.
     */
    void reshape(const std::vector<size_t>& new_dims);

};

#endif // DEVICEMEMORYIMPL_H
