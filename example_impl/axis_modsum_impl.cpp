#include "device_memory_impl.h"

#include "axis_modsum.h"

#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace lattica_hw_api {

template <typename T>
void axis_modsum(
    const std::shared_ptr<DeviceMemory<T>>& a,  // [m, s, k]
    const std::shared_ptr<DeviceMemory<T>>& p,  //       [k]
    std::shared_ptr<DeviceMemory<T>>& result    //    [m, k] (output)
) {
    if (a->dims.size() != 3 || p->dims.size() != 1 || result->dims.size() != 2) {
        throw std::invalid_argument("Input and output dimensions do not match the expected sizes.");
    }

    if (a->dims[0] != result->dims[0] || a->dims[2] != p->dims[0] || a->dims[2] != result->dims[1]) {
        throw std::invalid_argument("Mismatch in dimensions between inputs and output.");
    }

    size_t m = a->dims[0];
    size_t s = a->dims[1];
    size_t k = a->dims[2];

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            T sum = 0;
            for (size_t l = 0; l < s; ++l) {
                sum += a->at({i, l, j});
                sum %= p->at({j});
            }
            result->at({i, j}) = sum;
        }
    }
}
template void axis_modsum<int32_t>(
    const std::shared_ptr<DeviceMemory<int32_t>>& a,
    const std::shared_ptr<DeviceMemory<int32_t>>& p,
    std::shared_ptr<DeviceMemory<int32_t>>& result
);
template void axis_modsum<int64_t>(
    const std::shared_ptr<DeviceMemory<int64_t>>& a,
    const std::shared_ptr<DeviceMemory<int64_t>>& p,
    std::shared_ptr<DeviceMemory<int64_t>>& result
);

} // namespace lattica_hw_api
