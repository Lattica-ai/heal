#include "device_memory.h"
#include "device_tensor_ex_impl.h"
#include <stdexcept>

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
