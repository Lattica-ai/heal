#include "device_memory_impl.h"
#include "shape_manipulation.h"

#include <stdexcept>
#include <vector>
#include <cstddef>

namespace lattica_hw_api {

template <typename T>
void pad_single_axis(
    const std::shared_ptr<DeviceTensor<T>>& a,
    int64_t pad,
    int64_t axis,
    std::shared_ptr<DeviceTensor<T>>& result
) {
    if (!a) {
        throw std::invalid_argument("pad_single_axis: input tensor pointer is null");
    }
    if (!result) {
        throw std::invalid_argument("pad_single_axis: result tensor pointer is null");
    }

    // pad must be non-negative
    if (pad < 0) {
        throw std::invalid_argument("pad_single_axis: pad must be non-negative");
    }

    // grab input & output dims
    const auto& in_dims  = a->dims;
    const auto& out_dims = result->dims;
    const size_t rank    = in_dims.size();

    // check rank match
    if (out_dims.size() != rank) {
        throw std::invalid_argument("pad_single_axis: tensor ranks do not match");
    }

    // normalize axis (allow negative)
    if (axis < -static_cast<int64_t>(rank) || axis >= static_cast<int64_t>(rank)) {
        throw std::invalid_argument("pad_single_axis: axis index out of range");
    }
    int64_t norm_axis = axis < 0 ? axis + rank : axis;

    // verify output dimensions
    for (size_t i = 0; i < rank; ++i) {
        int64_t expected = (i == norm_axis) ? in_dims[i] + pad : in_dims[i];
        if (out_dims[i] != expected) {
            throw std::invalid_argument(
                "pad_single_axis: result tensor has incorrect dimension at axis "
                + std::to_string(i)
            );
        }
    }

    // compute total number of output elements
    size_t numel = 1;
    for (int64_t d : out_dims) {
        numel *= static_cast<size_t>(d);
    }

    // compute strides for output tensor (row-major)
    std::vector<size_t> strides(rank);
    strides[rank - 1] = 1;
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * static_cast<size_t>(out_dims[i + 1]);
    }

    // buffer for multi-dimensional index
    std::vector<int64_t> idx(rank);

    // iterate over every output element
    for (size_t lin = 0; lin < numel; ++lin) {
        // decode linear index → multi-index using precomputed strides
        size_t remaining = lin;
        for (size_t i = 0; i < rank; ++i) {
            idx[i]      = static_cast<int64_t>(remaining / strides[i]);
            remaining   = remaining % strides[i];
        }

        if (idx[norm_axis] < in_dims[norm_axis]) {
            // inside original tensor: copy value
            result->at(idx) = a->at(idx);
        } else {
            // in padded region: write zero
            result->at(idx) = static_cast<T>(0);
        }
    }
}

template void pad_single_axis<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>& a,
    int64_t pad,
    int64_t axis,
    std::shared_ptr<DeviceTensor<int32_t>>& result
);

template void pad_single_axis<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>& a,
    int64_t pad,
    int64_t axis,
    std::shared_ptr<DeviceTensor<int64_t>>& result
);

template void pad_single_axis<double>(
    const std::shared_ptr<DeviceTensor<double>>& a,
    int64_t pad,
    int64_t axis,
    std::shared_ptr<DeviceTensor<double>>& result
);

} // namespace lattica_hw_api
