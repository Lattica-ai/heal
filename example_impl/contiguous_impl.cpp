#include "device_memory_impl.h"
#include "contiguous.h"
#include <numeric>
#include <stdexcept>
#include <omp.h>

namespace lattica_hw_api {

    template <typename T>
    std::shared_ptr<DeviceTensor<T>> contiguous(const std::shared_ptr<DeviceTensor<T>>& tensor) {
        if (tensor->is_contiguous()) return tensor;

        int64_t total = std::accumulate(
            tensor->dims.begin(), tensor->dims.end(), int64_t(1), std::multiplies<>());

        std::shared_ptr<void> new_data = std::shared_ptr<void>(
            operator new(total * sizeof(T)),
            [](void* ptr) { operator delete(ptr); }
        );

        int64_t ndim = tensor->dims.size();
        T* dst_ptr = reinterpret_cast<T*>(new_data.get());

        // Compute strides for index-to-coord mapping
        std::vector<int64_t> shape = tensor->dims;
        std::vector<int64_t> flat_strides(ndim, 1);
        for (int64_t i = ndim - 2; i >= 0; --i) {
            flat_strides[i] = flat_strides[i + 1] * shape[i + 1];
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

        // Update tensor
        tensor->data = new_data;
        tensor->strides.resize(ndim);
        int64_t stride = 1;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            tensor->strides[i] = stride;
            stride *= tensor->dims[i];
        }

        return tensor;
    }

template std::shared_ptr<DeviceTensor<int32_t>> contiguous<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&);
template std::shared_ptr<DeviceTensor<int64_t>> contiguous<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&);
template std::shared_ptr<DeviceTensor<double>> contiguous<double>(const std::shared_ptr<DeviceTensor<double>>&);


} // namespace lattica_hw_api
