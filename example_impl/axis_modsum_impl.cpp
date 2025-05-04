#include "device_memory_impl.h"
#include "utils.h"
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

    const auto& out_shape = result->dims;
    const auto& in_shape = a->dims;
    const int64_t ndim = in_shape.size();
    const int64_t k_dim = p->dims[0];

    for (int64_t i = 0; i < k_dim; ++i) {
        if (p->at({i}) <= 0) {
          throw std::invalid_argument("Modulus value must be positive.");
        }
    }

    if (axis < 0 || axis >= ndim - 1) {
        throw std::invalid_argument("axis must be in range [0, ndim - 2] (can't reduce across last axis)");
    }

    if (in_shape.back() != k_dim) {
        throw std::invalid_argument("Last dimension of a must match shape of p");
    }

    std::vector<int64_t> expected_shape;
    expected_shape.reserve(ndim - 1);
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == axis) continue;
        expected_shape.push_back(in_shape[i]);
    }
    if (out_shape != expected_shape) {
        throw std::invalid_argument("Result tensor has wrong shape for the given axis");
    }

    int64_t result_numel = device_tensor_utils::numel(out_shape);
    const int64_t axis_size = in_shape[axis];

    #pragma omp parallel for
    for (int64_t flat_idx = 0; flat_idx < result_numel; ++flat_idx) {
        // Convert flat_idx to res_coord
        std::vector<int64_t> res_coord = device_tensor_utils::unravel_index(flat_idx, out_shape);

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
            T v = a->at(in_coord);
            v %= mod;
            if (v < 0) v += mod;
            sum += v;
            sum %= mod;
        }
        result->at(res_coord) = sum;


        result->at(res_coord) = sum;
    }
}


template <typename T>
void modmul_axis_sum(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& b,
    const std::shared_ptr<DeviceTensor<T>>& p,
    const std::shared_ptr<DeviceTensor<int64_t>>& perm,
    std::shared_ptr<DeviceTensor<T>>& result,
    bool apply_perm
) {
    // Validate modulus vector
    if (p->dims.size() != 1) {
        throw std::invalid_argument("p must be a 1D tensor of shape [k]");
    }
    const int64_t k_dim = p->dims[0];
    // Ensure all moduli are positive
    for (int64_t i = 0; i < k_dim; ++i) {
        if (p->at({i}) <= 0) {
            throw std::invalid_argument("Modulus value must be positive.");
        }
    }

    // Validate shapes
    const auto& in_shape = a->dims;
    if (in_shape.size() != 4) {
        throw std::invalid_argument("Tensor a must have shape [reps, n, sum_size, k].");
    }
    if (b->dims.size() != 3 || b->dims[0] != in_shape[1] || b->dims[1] != in_shape[2] || b->dims[2] != in_shape[3]) {
        throw std::invalid_argument("Tensor b must have shape [n, sum_size, k].");
    }
    if (result->dims.size() != 3 || result->dims[0] != in_shape[0] || result->dims[1] != in_shape[1] || result->dims[2] != in_shape[3]) {
        throw std::invalid_argument("Result tensor must have shape [reps, n, k].");
    }

    // Validate permutation
    const int64_t sum_axis = 2;
    const int64_t perm_axis = 1;
    if (apply_perm) {
        // perm must be a 1D tensor matching the size of the axis before the sum axis
        if (perm->dims.size() != 1 || perm->dims[0] != in_shape[perm_axis]) {
            throw std::invalid_argument("perm must be a 1D tensor matching the size of the axis before the sum axis");
        }
        // ensure each perm index is within [0, n)
        for (int64_t i = 0; i < perm->dims[0]; ++i) {
            int64_t idx = perm->at({i});
            if (idx < 0 || idx >= in_shape[perm_axis]) {
                throw std::invalid_argument("perm index out of bounds");
            }
        }
    }

    // Prepare looping
    const auto& out_shape = result->dims;
    const int64_t result_numel = device_tensor_utils::numel(out_shape);
    const int64_t sum_size = in_shape[sum_axis];

    #pragma omp parallel for
    for (int64_t flat_idx = 0; flat_idx < result_numel; ++flat_idx) {
        // Coordinates in output: [rep, n, k]
        std::vector<int64_t> res_coord = device_tensor_utils::unravel_index(flat_idx, out_shape);
        int64_t rep = res_coord[0];
        int64_t n   = res_coord[1];
        int64_t c   = res_coord[2];

        // Determine b's batch index
        int64_t b_n = apply_perm ? perm->at({n}) : n;

        T total = 0;
        for (int64_t s = 0; s < sum_size; ++s) {
            T aval = a->at({rep, n, s, c});
            T bval = b->at({b_n, s, c});
            T mod = p->at({c});

            // Multiply and normalize
            T prod = aval * bval;
            prod = prod % mod;
            if (prod < 0) prod += mod;

            // Accumulate and normalize
            T acc = total + prod;
            total = acc % mod;
        }

        result->at({rep, n, c}) = total;
    }
}

// Explicit template instantiations

template void axis_modsum<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    std::shared_ptr<DeviceTensor<int32_t>>&,
    int64_t
);

template void axis_modsum<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t
);

template void modmul_axis_sum<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    std::shared_ptr<DeviceTensor<int32_t>>&,
    bool
);

template void modmul_axis_sum<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    std::shared_ptr<DeviceTensor<int64_t>>&,
    bool
);

} // namespace lattica_hw_api
