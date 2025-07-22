#include "contiguous.h"
#include "device_tensor_ex_impl.h"
#include <stdexcept>

namespace lattica_hw_api {

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

template std::shared_ptr<DeviceTensor<int8_t>> contiguous<int8_t>(const std::shared_ptr<DeviceTensor<int8_t>>&);
template std::shared_ptr<DeviceTensor<int32_t>> contiguous<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&);
template std::shared_ptr<DeviceTensor<int64_t>> contiguous<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&);


} // namespace lattica_hw_api
