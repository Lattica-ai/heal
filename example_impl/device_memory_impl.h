#ifndef DeviceTensorIMPL_H
#define DeviceTensorIMPL_H

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

    static DeviceTensor<T> slice_view(const std::shared_ptr<DeviceTensor<T>>& base,
                                                    std::vector<int64_t> new_dims,
                                                    std::vector<int64_t> new_strides,
                                                    int64_t offset_in_elements);

private:
    DeviceTensor(std::vector<int64_t> dims,
        std::vector<int64_t> strides,
        std::shared_ptr<void> alias_data);
};

#endif // DeviceTensorIMPL_H
