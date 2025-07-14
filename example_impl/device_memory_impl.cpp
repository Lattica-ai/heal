#include "device_memory_impl.h"
#include <iostream>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <functional>
#include <torch/torch.h>

// Explicit template instantiations
template struct DeviceTensor<int8_t>;
template struct DeviceTensor<int32_t>;
template struct DeviceTensor<int64_t>;

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

namespace lattica_hw_api {

template <typename T>
std::shared_ptr<DeviceTensor<T>> zeros(const std::vector<int64_t>& dims) {
    int64_t total_elems = std::accumulate(dims.begin(), dims.end(), int64_t(1), std::multiplies<int64_t>());
    void* buffer = calloc(total_elems, sizeof(T));
    std::vector<int64_t> strides(dims.size());
    int64_t stride = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
    }
    return std::make_shared<DeviceTensor<T>>(dims, strides, buffer);
}

template <typename T>
std::shared_ptr<DeviceTensor<T>> empty(const std::vector<int64_t>& dims) {
    int64_t total_elems = std::accumulate(dims.begin(), dims.end(), int64_t(1), std::multiplies<int64_t>());
    void* buffer = malloc(total_elems * sizeof(T));
    std::vector<int64_t> strides(dims.size());
    int64_t stride = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
    }
    return std::make_shared<DeviceTensor<T>>(dims, strides, buffer);
}

template <typename T>
std::shared_ptr<DeviceTensor<T>> host_to_device(const torch::Tensor& tensor) {
    if (tensor.scalar_type() != torch::CppTypeToScalarType<T>()) {
        throw std::runtime_error("Tensor dtype does not match template parameter T.");
    }

    std::vector<int64_t> dims(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int64_t> strides(tensor.strides().begin(), tensor.strides().end());
    return std::make_shared<DeviceTensor<T>>(dims, strides, tensor.data_ptr());
}

template <typename T>
torch::Tensor device_to_host(const std::shared_ptr<DeviceTensor<T>>& memory) {
    auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>());
    return torch::from_blob(
        memory->data.get(),
        memory->dims,
        memory->strides,
        [](void*) {},  // no-op deleter since memory is owned by shared_ptr
        options
    ).clone();  // clone to detach from external buffer if needed
}

// Explicit instantiations
template std::shared_ptr<DeviceTensor<int32_t>> zeros<int32_t>(const std::vector<int64_t>&);
template std::shared_ptr<DeviceTensor<int64_t>> zeros<int64_t>(const std::vector<int64_t>&);

template std::shared_ptr<DeviceTensor<int8_t>> empty<int8_t>(const std::vector<int64_t>&);
template std::shared_ptr<DeviceTensor<int32_t>> empty<int32_t>(const std::vector<int64_t>&);
template std::shared_ptr<DeviceTensor<int64_t>> empty<int64_t>(const std::vector<int64_t>&);

template std::shared_ptr<DeviceTensor<int8_t>> host_to_device<int8_t>(const torch::Tensor&);
template std::shared_ptr<DeviceTensor<int32_t>> host_to_device<int32_t>(const torch::Tensor&);
template std::shared_ptr<DeviceTensor<int64_t>> host_to_device<int64_t>(const torch::Tensor&);

template torch::Tensor device_to_host<int8_t>(const std::shared_ptr<DeviceTensor<int8_t>>&);
template torch::Tensor device_to_host<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&);
template torch::Tensor device_to_host<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&);

} // namespace lattica_hw_api
