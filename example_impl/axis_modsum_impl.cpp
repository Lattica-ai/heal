#include "device_memory_impl.h"

#include "axis_modsum.h"

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <omp.h>

namespace lattica_hw_api {

template <typename T>
void axis_modsum(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& p,
    std::shared_ptr<DeviceTensor<T>>& result,
    int64_t axis
) {
    if (p->dims.size() != 1) {
        throw std::invalid_argument("p must be a 1D tensor of shape [k]");
    }

    const auto& in_shape = a->dims;
    const int64_t ndim = in_shape.size();
    const int64_t k_dim = p->dims[0];

    if (axis < 0 || axis >= ndim - 1) {
        throw std::invalid_argument("axis must be in range [0, ndim - 2] (can't reduce across last axis)");
    }

    if (in_shape.back() != k_dim) {
        throw std::invalid_argument("Last dimension of a must match shape of p");
    }

    int64_t result_numel = 1;
    for (auto d : result->dims) result_numel *= d;
    const int64_t axis_size = in_shape[axis];

    // Compute flat-to-multidim strides for result
    std::vector<int64_t> res_strides(result->dims.size(), 1);
    for (int64_t i = result->dims.size() - 2; i >= 0; --i) {
        res_strides[i] = res_strides[i + 1] * result->dims[i + 1];
    }

    #pragma omp parallel for
    for (int64_t flat_idx = 0; flat_idx < result_numel; ++flat_idx) {
        // Convert flat_idx to res_coord
        std::vector<int64_t> res_coord(result->dims.size());
        int64_t rem = flat_idx;
        for (int64_t i = 0; i < (int64_t)res_coord.size(); ++i) {
            res_coord[i] = rem / res_strides[i];
            rem %= res_strides[i];
        }

        // Build input coord with axis inserted
        std::vector<int64_t> in_coord;
        in_coord.reserve(ndim);
        for (int64_t i = 0, j = 0; i < ndim; ++i) {
            if (i == axis) {
                in_coord.push_back(0); // placeholder
            } else {
                in_coord.push_back(res_coord[j++]);
            }
        }

        T mod = p->at({in_coord[ndim - 1]});
        T sum = 0;
        for (int64_t r = 0; r < axis_size; ++r) {
            in_coord[axis] = r;
            sum = (sum + a->at(in_coord)) % mod;
        }

        result->at(res_coord) = sum;
    }

}


template void axis_modsum<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>& a,
    const std::shared_ptr<DeviceTensor<int32_t>>& p,
    std::shared_ptr<DeviceTensor<int32_t>>& result,
    int64_t axis
);

template void axis_modsum<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>& a,
    const std::shared_ptr<DeviceTensor<int64_t>>& p,
    std::shared_ptr<DeviceTensor<int64_t>>& result,
    int64_t axis
);

} // namespace lattica_hw_api
