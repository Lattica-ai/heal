#include "device_memory_impl.h"
#include "take_along_axis.h"
#include <stdexcept>

namespace lattica_hw_api {

template <typename T>
void take_along_axis(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<int64_t>>& indices,
    int64_t axis,
    std::shared_ptr<DeviceTensor<T>>& result
) {
    const size_t rank = a->dims.size();

    // Normalize & validate axis
    if (axis < -static_cast<int64_t>(rank) || axis >= static_cast<int64_t>(rank)) {
        throw std::out_of_range("Axis out of range");
    }
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }

    // Rank‐match check
    if (indices->dims.size() != rank) {
        throw std::invalid_argument("Indices tensor rank must match input rank");
    }

    // Validate indices shape against input shape
    for (size_t i = 0; i < rank; ++i) {
        if (i != static_cast<size_t>(axis) && indices->dims[i] != a->dims[i]) {
            throw std::invalid_argument(
                "take_along_axis: indices shape must match input shape on all dims except axis. "
                "Got input shape = " + std::to_string(a->dims[i]) +
                ", indices shape = " + std::to_string(indices->dims[i]) +
                " at axis " + std::to_string(i));
        }
    }

    // Compute number of output elements from indices shape
    int64_t total = 1;
    for (size_t i = 0; i < rank; ++i) {
        total *= indices->dims[i];
    }

    // Precompute row-major strides for unraveling a flat output index
    std::vector<int64_t> out_strides(rank, 1);
    for (int64_t i = rank - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * indices->dims[i + 1];
    }

    // Pointers into your flat buffers
    T* a_data = static_cast<T*>(a->data.get());
    int64_t* idx_data = static_cast<int64_t*>(indices->data.get());
    T* out_data = static_cast<T*>(result->data.get());

    std::vector<int64_t> idx_full(rank), idx_src(rank);

    // Iterate over every output element in flat (1D) index space
    for (int64_t flat = 0; flat < total; ++flat) {
        int64_t rem = flat;

        // Convert the flat output index into a multi-dimensional index (row-major unraveling) using precomputed strides
        for (size_t i = 0; i < rank; ++i) {
            idx_full[i] = rem / out_strides[i];
            rem %= out_strides[i];
        }

        // Fetch the “select” index from the indices buffer at this multi-dimensional coordinate
        int64_t idx_offset = 0;
        for (size_t i = 0; i < rank; ++i) {
            idx_offset += idx_full[i] * indices->strides[i];
        }
        int64_t sel = idx_data[idx_offset];
        const int64_t axis_size = a->dims[axis];

        // Validate the selected index against the size of the axis dimension
        if (sel < 0) sel += axis_size;
        if (sel < 0 || sel >= axis_size) {
            throw std::out_of_range("Index out of bounds in take_along_axis");
        }

        // Build the source coordinate in `a` by copying idx_full and
        //     replacing the value along the `axis` dimension with `sel`
        idx_src = idx_full;
        idx_src[axis] = sel;

        // Compute the flattened offset in `a` corresponding to this coordinate
        int64_t src_offset = 0;
        for (size_t i = 0; i < rank; ++i) {
            src_offset += idx_src[i] * a->strides[i];
        }

        // Write the gathered value into the result tensor at position `flat`
        out_data[flat] = a_data[src_offset];
    }
}


template void take_along_axis<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<int32_t>>&
);

template void take_along_axis<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<int64_t>>&
);

} // namespace lattica_hw_api
