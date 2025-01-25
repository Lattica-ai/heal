#include "device_memory_impl.h"
#include "permute.h"
#include <stdexcept>
#include <iostream>

namespace lattica_hw_api {

template <typename T>
void permute(
    const std::shared_ptr<DeviceMemory<T>>& a,         // [l, m, k]
    const std::shared_ptr<DeviceMemory<T>>& perms,    // [l, m]
    std::shared_ptr<DeviceMemory<T>>& result          // [l, m, k] (output)
) {
    // Validate input dimensions
    if (a->dimensions != 3 || perms->dimensions != 2 || result->dimensions != 3) {
        throw std::invalid_argument("Invalid dimensions. Expected [l, m, k] for 'a', [l, m] for 'perms', and [l, m, k] for 'result'.");
    }

    size_t l = a->dims[0];
    size_t m = a->dims[1];
    size_t k = a->dims[2];

    if (perms->dims[0] != l || perms->dims[1] != m || result->dims[0] != l || result->dims[1] != m || result->dims[2] != k) {
        throw std::invalid_argument("Dimension mismatch between 'a', 'perms', and 'result'.");
    }

    // Apply permutations
    for (size_t i = 0; i < l; ++i) {
        for (size_t j = 0; j < m; ++j) {
            size_t perm_index = perms->at({i, j});
            if (perm_index >= m) {
                throw std::out_of_range("Permutation index out of bounds.");
            }

            for (size_t z = 0; z < k; ++z) {
                result->at({i, j, z}) = a->at({i, perm_index, z});
            }
        }
    }
}
template void permute<int32_t>(const std::shared_ptr<DeviceMemory<int32_t>>&, const std::shared_ptr<DeviceMemory<int32_t>>&, std::shared_ptr<DeviceMemory<int32_t>>&);
template void permute<int64_t>(const std::shared_ptr<DeviceMemory<int64_t>>&, const std::shared_ptr<DeviceMemory<int64_t>>&, std::shared_ptr<DeviceMemory<int64_t>>&);

} // namespace lattica_hw_api
