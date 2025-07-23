#include "memory_management.h"
#include "device_tensor_ex.h"
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
std::shared_ptr<DeviceTensor<T>> contiguous(const std::shared_ptr<DeviceTensor<T>>& tensor) {
    if (tensor->is_contiguous()) return tensor;

    auto& dims = tensor->dims;
    int64_t total = 1;
    for (int64_t d : dims) total *= d;

    std::shared_ptr<void> new_data = std::shared_ptr<void>(
        operator new(total * sizeof(T)),
        [](void* ptr) { operator delete(ptr); }
    );

    int64_t ndim = dims.size();
    T* dst_ptr = reinterpret_cast<T*>(new_data.get());

    // Compute strides for index-to-coord mapping
    std::vector<int64_t> flat_strides(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; --i) {
        flat_strides[i] = flat_strides[i + 1] * dims[i + 1];
    }

    #pragma omp parallel for
    for (int64_t idx = 0; idx < total; ++idx) {
        std::vector<int64_t> coord(ndim);
        int64_t remaining = idx;
        for (int64_t d = 0; d < ndim; ++d) {
            coord[d] = remaining / flat_strides[d];
            remaining %= flat_strides[d];
        }
        dst_ptr[idx] = tensor->at(coord);
    }

    // Compute contiguous strides
    std::vector<int64_t> new_strides(ndim, 1);
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= tensor->dims[i];
    }

    return std::make_shared<DeviceTensor<T>>(tensor->dims, new_strides, new_data);
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

template std::shared_ptr<DeviceTensor<int8_t>> contiguous<int8_t>(const std::shared_ptr<DeviceTensor<int8_t>>&);
template std::shared_ptr<DeviceTensor<int32_t>> contiguous<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&);
template std::shared_ptr<DeviceTensor<int64_t>> contiguous<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&);

template std::shared_ptr<DeviceTensor<int8_t>> host_to_device<int8_t>(const torch::Tensor&);
template std::shared_ptr<DeviceTensor<int32_t>> host_to_device<int32_t>(const torch::Tensor&);
template std::shared_ptr<DeviceTensor<int64_t>> host_to_device<int64_t>(const torch::Tensor&);

template torch::Tensor device_to_host<int8_t>(const std::shared_ptr<DeviceTensor<int8_t>>&);
template torch::Tensor device_to_host<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&);
template torch::Tensor device_to_host<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&);

} // namespace lattica_hw_api
