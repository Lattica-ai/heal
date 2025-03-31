#include "device_memory_impl.h"
#include "permute.h"
#include <stdexcept>
#include <iostream>

namespace lattica_hw_api {

template <typename T>
void permute(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& perms,
    std::shared_ptr<DeviceTensor<T>>& result,
    int64_t elementwise_axis,
    int64_t perm_axis
) {
    const auto& shape = a->dims;
    const int64_t ndim = shape.size();

    if (result->dims != shape) {
        throw std::invalid_argument("Result shape must match input shape.");
    }

    if (elementwise_axis < 0 || elementwise_axis >= ndim || perm_axis < 0 || perm_axis >= ndim) {
        throw std::invalid_argument("Axis out of bounds.");
    }
    if (elementwise_axis == perm_axis) {
        throw std::invalid_argument("elementwise_axis and perm_axis must be different.");
    }

    const int64_t l = shape[elementwise_axis];
    const int64_t m = shape[perm_axis];

    if (perms->dims.size() != 2 || perms->dims[0] != l || perms->dims[1] != m) {
        throw std::invalid_argument("Perms must have shape [l, m] where l and m match a.shape at elementwise and perm axes.");
    }

    // Generate coordinate iterator over all batch dimensions
    std::vector<int64_t> coord(ndim, 0);

    while (true) {
        // Read indices for current (batch slice, l, m)
        int64_t l_idx = coord[elementwise_axis];
        int64_t m_idx = coord[perm_axis];
        int64_t perm_idx = perms->at({l_idx, m_idx});
        if (perm_idx >= m) {
            throw std::out_of_range("Permutation index out of bounds.");
        }

        // Build source coordinate
        std::vector<int64_t> src_coord = coord;
        src_coord[perm_axis] = perm_idx;

        result->at(coord) = a->at(src_coord);

        // Advance coordinate
        int64_t dim = ndim - 1;
        while (dim >= 0) {
            coord[dim]++;
            if (coord[dim] < shape[dim]) break;
            coord[dim] = 0;
            dim--;
        }
        if (dim < 0) break;
    }
}

template void permute<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>& a,
    const std::shared_ptr<DeviceTensor<int32_t>>& perms,
    std::shared_ptr<DeviceTensor<int32_t>>& result,
    int64_t elementwise_axis,
    int64_t perm_axis
);

template void permute<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>& a,
    const std::shared_ptr<DeviceTensor<int64_t>>& perms,
    std::shared_ptr<DeviceTensor<int64_t>>& result,
    int64_t elementwise_axis,
    int64_t perm_axis
);

} // namespace lattica_hw_api
