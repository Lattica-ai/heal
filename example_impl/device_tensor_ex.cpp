#include "device_tensor_ex.h"
#include <iostream>
#include <cstring>
#include <functional>

template <typename T>
DeviceTensor<T>::DeviceTensor(const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& strides,
                 const void* src_data) : dims(dims), strides(strides)
{
    int64_t total_bytes = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        total_bytes += (dims[i] - 1) * strides[i];
    }
    total_bytes *= sizeof(T);

    void* buffer = malloc(total_bytes);
    if (!buffer) throw std::bad_alloc();
    std::memcpy(buffer, src_data, total_bytes);
    data.reset(buffer, free);
}

template<typename T>
DeviceTensor<T>::DeviceTensor(
    std::vector<int64_t> dims,
    std::vector<int64_t> strides,
    std::shared_ptr<void> alias_data)
    : dims(std::move(dims)), strides(std::move(strides)), data(std::move(alias_data)) {}

template <typename T>
bool DeviceTensor<T>::is_contiguous() const {
    int64_t expected_stride = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
        if (dims[i] == 1) continue;  // skip singleton dims
        if (strides[i] != expected_stride) return false;
        expected_stride *= dims[i];
    }
    return true;
}

template <typename T>
T& DeviceTensor<T>::at(const std::vector<int64_t>& indices) {
    return const_cast<T&>(static_cast<const DeviceTensor<T>&>(*this).at(indices));
}

template <typename T>
const T& DeviceTensor<T>::at(const std::vector<int64_t>& indices) const {
    if (indices.size() != dims.size()) {
        std::cout << indices.size() << "     " << dims.size() << std::endl;
        throw std::invalid_argument("Number of indices does not match tensor dimensions.");
    }

    int64_t offset = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (indices[i] >= dims[i]) {
            throw std::out_of_range("Index out of bounds.");
        }
        offset += indices[i] * strides[i];
    }

    return reinterpret_cast<T*>(data.get())[offset];
}


template <typename T>
T& DeviceTensor<T>::at_with_broadcast(const std::vector<int64_t>& full_indices) {
    return const_cast<T&>(static_cast<const DeviceTensor<T>&>(*this).at_with_broadcast(full_indices));
}

template <typename T>
const T& DeviceTensor<T>::at_with_broadcast(const std::vector<int64_t>& full_indices) const {
    std::vector<int64_t> adjusted;
    int64_t offset = full_indices.size() - dims.size();
    for (size_t i = 0; i < dims.size(); ++i) {
        adjusted.push_back(dims[i] == 1 ? 0 : full_indices[i + offset]);
    }
    return at(adjusted);
}

template <typename T>
void DeviceTensor<T>::print() const {
    std::cout << "DeviceTensor<" << typeid(T).name() << "> ";
    std::cout << "Shape: [";
    for (auto d : dims) std::cout << d << " ";
    std::cout << "]  Strides: [";
    for (auto s : strides) std::cout << s << " ";
    std::cout << "]\n";
    std::cout << "]\nData: ";

    std::vector<int64_t> idx(dims.size(), 0);
    std::function<void(int64_t)> recurse = [&](int64_t dim) {
        if (dim == dims.size()) {
            std::cout << at(idx);
            return;
        }

        std::cout << "{";
        for (int64_t i = 0; i < dims[dim]; ++i) {
            idx[dim] = i;
            if (i > 0) std::cout << ", ";
            recurse(dim + 1);
        }
        std::cout << "}";
    };

    recurse(0);
    std::cout << "\n\n";
}

template <typename T>
void DeviceTensor<T>::print_metadata() const {
    std::cout << "DeviceTensor<" << typeid(T).name() << "> ";
    std::cout << "Shape: [";
    for (auto d : dims) std::cout << d << " ";
    std::cout << "]  Strides: [";
    for (auto s : strides) std::cout << s << " ";
    std::cout << "]\n\n";
}

template <typename T>
int64_t DeviceTensor<T>::numel() const
{
    int64_t total = 1;
    for (auto d : dims) {
        total *= d;
    }
    return total;
}

template <typename T>
std::vector<int64_t> DeviceTensor<T>::compute_contiguous_strides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

// Maps a flat index to a coordinate for a given shape and stride
template <typename T>
std::vector<int64_t> DeviceTensor<T>::unravel_index(int64_t flat_idx,
                                                    const std::vector<int64_t>& shape,
                                                    const std::vector<int64_t>& strides)
{
    std::vector<int64_t> coord(shape.size());
    int64_t remaining = flat_idx;
    for (size_t i = 0; i < shape.size(); ++i) {
        coord[i] = remaining / strides[i];
        remaining %= strides[i];
    }
    return coord;
}


// Explicit template instantiations
template class DeviceTensor<int8_t>;
template class DeviceTensor<int32_t>;
template class DeviceTensor<int64_t>;