#pragma once

#include <vector>
#include <memory>

/**
 * @brief A class to manage multi-dimensional memory buffers.
 * This is the implementation of the public DeviceTensor API.
 */
template <typename T>
class DeviceTensor {
public:
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::shared_ptr<void> data;

    DeviceTensor(std::vector<int64_t> dims,
        std::vector<int64_t> strides,
        std::shared_ptr<void> alias_data);
    DeviceTensor(const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& strides,
                 const void* src_data);

    void print() const;
    void print_metadata() const;
    bool is_contiguous() const;
    int64_t numel() const;

    // Element access
    T& at(const std::vector<int64_t>& indices);
    const T& at(const std::vector<int64_t>& indices) const;

    // Broadcast-aware access
    T& at_with_broadcast(const std::vector<int64_t>& full_indices);
    const T& at_with_broadcast(const std::vector<int64_t>& full_indices) const;

    static std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape);
    static std::vector<int64_t> unravel_index(int64_t flat_idx,
                                              const std::vector<int64_t>& shape,
                                              const std::vector<int64_t>& strides);
};
