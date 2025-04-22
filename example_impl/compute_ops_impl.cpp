#include "device_memory_impl.h"
#include "compute_ops.h"
#include <stdexcept>

namespace lattica_hw_api {

    template <typename T>
    void take_along_axis(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<IndexType>>& indices,
        IndexType axis,
        std::shared_ptr<DeviceTensor<T>>& result
    ) {
        const size_t rank = a->dims.size();

        // 1) Normalize & validate axis
        if (axis < -static_cast<IndexType>(rank) || axis >= static_cast<IndexType>(rank)) {
            throw std::out_of_range("Axis out of range");
        }
        if (axis < 0) {
            axis += static_cast<IndexType>(rank);
        }

        // 2) Rank‐match check
        if (indices->dims.size() != rank) {
            throw std::invalid_argument("Indices tensor rank must match input rank");
        }

        // 3) Shape‐check on non‐axis dims
        for (size_t i = 0; i < rank; ++i) {
            if (i != static_cast<size_t>(axis) && a->dims[i] != indices->dims[i]) {
                throw std::invalid_argument("Shape mismatch at non-axis dimension");
            }
        }

        // 4) Compute number of output elements from indices shape
        int64_t total = 1;
        for (size_t i = 0; i < rank; ++i) {
            total *= indices->dims[i];
        }

        // 5) Build row‑major strides for unraveling an index into the indices‑shape
        std::vector<int64_t> out_strides(rank);
        int64_t s = 1;
        for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
            out_strides[i] = s;
            s *= indices->dims[i];
        }

        // 6) Pointers into your flat buffers
        T*          a_data    = static_cast<T*>(a->data.get());
        IndexType*  idx_data  = static_cast<IndexType*>(indices->data.get());
        T*          out_data  = static_cast<T*>(result->data.get());

        std::vector<int64_t> idx_full(rank), idx_src(rank);

        // 7) Iterate over every output element in flat (1D) index space
        for (int64_t flat = 0; flat < total; ++flat) {
            int64_t rem = flat;

            // 8) Convert the flat output index into a multi-dimensional index (row-major unraveling)
            //    This gives us the coordinate in the `indices` tensor that we're currently evaluating.
            for (size_t i = 0; i < rank; ++i) {
                idx_full[i] = rem / out_strides[i];  // compute index in dimension i
                rem %= out_strides[i];
            }

            // 9) Fetch the “select” index from the indices buffer at this multi-dimensional coordinate
            int64_t idx_offset = 0;
            for (size_t i = 0; i < rank; ++i) {
                idx_offset += idx_full[i] * indices->strides[i];
            }
            IndexType sel = idx_data[idx_offset];

            // 10) Handle negative indices (Python-style)
            if (sel < 0) {
                sel += static_cast<IndexType>(a->dims[axis]);
            }
            if (sel < 0 || sel >= static_cast<IndexType>(a->dims[axis])) {
                throw std::out_of_range("Index out of range");
            }

            // 11) Build the source coordinate in `a` by copying idx_full and
            //     replacing the value along the `axis` dimension with `sel`
            for (size_t i = 0; i < rank; ++i) {
                idx_src[i] = idx_full[i];
            }
            idx_src[axis] = sel;

            // 12) Compute the flattened offset in `a` corresponding to this coordinate
            int64_t src_offset = 0;
            for (size_t i = 0; i < rank; ++i) {
                src_offset += idx_src[i] * a->strides[i];
            }

            // 13) Write the gathered value into the result tensor at position `flat`
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

template void take_along_axis<float>(
    const std::shared_ptr<DeviceTensor<float>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<float>>&
);

template void take_along_axis<double>(
    const std::shared_ptr<DeviceTensor<double>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<double>>&
);

}