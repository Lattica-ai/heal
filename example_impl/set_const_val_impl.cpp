#include "device_memory_impl.h"
#include "set_const_val.h"
#include <stdexcept>

namespace lattica_hw_api {

template <typename T>
void set_const_val(
    const std::shared_ptr<DeviceTensor<T>>& a,
    T val
) {
    // must have a valid tensor
    if (!a) {
        throw std::invalid_argument("set_const_val: input tensor is null");
    }

    const auto& dims = a->dims;
    const size_t rank = dims.size();
    int64_t numel = 1;
    for (int64_t d : dims) numel *= d;

    // Compute strides for row-major order
    std::vector<int64_t> strides(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Buffer for multidimensional index
    std::vector<int64_t> idx(rank);

    // Iterate every element by converting linear index → multi-index via strides
    for (int64_t lin = 0; lin < numel; ++lin) {
        int64_t rem = lin;
        for (int64_t i = 0; i < (int64_t)rank; ++i) {
            idx[i] = rem / strides[i];
            rem %= strides[i];
        }
        // set to constant
        a->at(idx) = val;
    }
}


template <typename T>
void pad_single_axis(
    const std::shared_ptr<DeviceTensor<T>>& a,
    int64_t pad,
    int64_t axis,
    std::shared_ptr<DeviceTensor<T>>& result
) {
    // pad must be non-negative
    if (pad < 0) {
        throw std::invalid_argument("pad_single_axis: pad must be non-negative");
    }

    // grab input & output dims
    const auto& in_dims = a->dims;
    const auto& out_dims = result->dims;
    const int64_t rank = in_dims.size();

    // check rank match
    if (out_dims.size() != rank) {
        throw std::invalid_argument("pad_single_axis: tensor ranks do not match");
    }

    // normalize axis (allow negative)
    axis = axis < 0 ? axis + rank : axis;
    if ((axis < 0) || (axis >= rank)) {
        throw std::invalid_argument("pad_single_axis: axis index out of range");
    }

    // verify output dimensions
    for (size_t i = 0; i < rank; ++i) {
        int64_t expected = (i == axis) ? in_dims[i] + pad : in_dims[i];
        if (out_dims[i] != expected) {
            throw std::invalid_argument(
                "pad_single_axis: result tensor has incorrect dimension at axis "
                + std::to_string(i)
            );
        }
    }

    // compute total number of output elements
    int64_t numel = 1;
    for (int64_t d : out_dims) numel *= d;

    // compute strides for output tensor (row-major)
    std::vector<int64_t> strides(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * out_dims[i + 1];

    // iterate over every output element
    for (int64_t flat = 0; flat < numel; ++flat) {
        // decode linear index → multi-index using precomputed strides
        std::vector<int64_t> coord(rank);
        int64_t rem = flat;
        for (int64_t i = 0; i < rank; ++i) {
            coord[i] = rem / strides[i];
            rem %= strides[i];
        }

        if (coord[axis] < in_dims[axis]) {
            // inside original tensor: copy value
            result->at(coord) = a->at(coord);
        } else {
            // in padded region: write zero
            result->at(coord) = static_cast<T>(0);
        }
    }
}


template void set_const_val<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&, int32_t);
template void set_const_val<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&, int64_t);
template void pad_single_axis<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&, int64_t, int64_t, std::shared_ptr<DeviceTensor<int32_t>>&);
template void pad_single_axis<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&, int64_t, int64_t, std::shared_ptr<DeviceTensor<int64_t>>&);

} // namespace lattica_hw_api