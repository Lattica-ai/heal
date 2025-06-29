#ifndef DeviceTensorIMPL_H
#define DeviceTensorIMPL_H

#include <vector>
#include <memory>

/**
 * @brief A structure to manage multi-dimensional memory buffers.
 * This is the implementation of the public DeviceTensor API.
 */
template <typename T>
struct DeviceTensor {
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::shared_ptr<void> data;

    DeviceTensor(const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& strides,
                 const void* src_data);

    void reshape(const std::vector<int64_t>& new_dims);
    void print() const;
    void print_metadata() const;
    bool is_contiguous() const;


    // Element access
    T& at(const std::vector<int64_t>& indices);
    const T& at(const std::vector<int64_t>& indices) const;

    // Broadcast-aware access
    T& at_with_broadcast(const std::vector<int64_t>& full_indices);
    const T& at_with_broadcast(const std::vector<int64_t>& full_indices) const;
};

#endif // DeviceTensorIMPL_H
