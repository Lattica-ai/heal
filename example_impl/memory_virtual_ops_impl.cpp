#include "device_memory_impl.h"
#include "memory_virtual_ops.h"
#include <stdexcept>

namespace lattica_hw_api {

    template <typename T>
    std::shared_ptr<DeviceTensor<T>> expand(
        const std::shared_ptr<DeviceTensor<T>>& a,
        int64_t axis,
        int64_t repeats
    ) {
        if (repeats <= 0) {
            throw std::invalid_argument("Repeat count must be positive.");
        }

        int64_t ndim = static_cast<int64_t>(a->dims.size());
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis >= ndim) {
            throw std::invalid_argument("Invalid repeat dimension.");
        }

        if (a->dims[axis] != 1) {
            throw std::invalid_argument("Can only expand a dimension of size 1.");
        }

        // New shape: a->dims with axis scaled by repeats
        std::vector<int64_t> new_dims = a->dims;
        new_dims[axis] *= repeats;

        // New strides: same as a, but axis stride becomes 0
        std::vector<int64_t> new_strides = a->strides;
        new_strides[axis] = 0;

        // Share the underlying data pointer, just modify metadata
        a->dims = new_dims;
        a->strides = new_strides;

        return a;
    }

    template <typename T>
    std::shared_ptr<DeviceTensor<T>> squeeze(
        const std::shared_ptr<DeviceTensor<T>>& a,
        int64_t axis
    ) {
        int64_t ndim = static_cast<int64_t>(a->dims.size());
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis >= ndim) {
            throw std::invalid_argument("Invalid squeeze dimension.");
        }

        if (a->dims[axis] != 1) {
            throw std::invalid_argument("Can only squeeze dimensions of size 1.");
        }

        std::vector<int64_t> new_dims = a->dims;
        std::vector<int64_t> new_strides = a->strides;
        new_dims.erase(new_dims.begin() + axis);
        new_strides.erase(new_strides.begin() + axis);

        a->dims = new_dims;
        a->strides = new_strides;
        return a;
    }

    template <typename T>
    std::shared_ptr<DeviceTensor<T>> unsqueeze(
        const std::shared_ptr<DeviceTensor<T>>& a,
        int64_t axis
    ) {
        int64_t ndim = static_cast<int64_t>(a->dims.size());
        if (axis < 0) axis += (ndim + 1);
        if (axis < 0 || axis > ndim) {
            throw std::invalid_argument("Invalid unsqueeze dimension.");
        }

        std::vector<int64_t> new_dims = a->dims;
        std::vector<int64_t> new_strides = a->strides;
        new_dims.insert(new_dims.begin() + axis, 1);
        new_strides.insert(new_strides.begin() + axis, 0);

        a->dims = new_dims;
        a->strides = new_strides;
        return a;
    }

    template<typename T>
    std::shared_ptr<DeviceTensor<T>> get_slice(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::vector<SliceArg>& slices
    ) {
        size_t rank = a->dims.size();
        if (slices.size() != rank) {
            throw std::invalid_argument(
                "get_slice: number of SliceArg entries ("
                + std::to_string(slices.size())
                + ") must equal tensor rank ("
                + std::to_string(rank) + ")."
            );
        }

        struct AxisInfo {
            bool is_index;   // true if we hold int64_t
            int64_t index;      // valid if is_index==true
            int64_t start;      // valid if is_index==false
            int64_t end;        // valid if is_index==false (exclusive)
            int64_t step;       // valid if is_index==false (>0)
        };

        std::vector<AxisInfo> infos(rank);
        for (size_t dim = 0; dim < rank; ++dim) {
            if (std::holds_alternative<int64_t>(slices[dim])) {
                // Single‐index case
                int64_t idx = std::get<int64_t>(slices[dim]);
                if (idx < 0 || idx >= a->dims[dim]) {
                    throw std::out_of_range(
                        "get_slice: index " + std::to_string(idx)
                        + " out of range for dim " + std::to_string(dim)
                        + " (size=" + std::to_string(a->dims[dim]) + ")"
                    );
                }
                infos[dim].is_index = true;
                infos[dim].index    = idx;
            }
            else {
                // Slice case
                const Slice& s = std::get<Slice>(slices[dim]);
                // Validate: 0 ≤ start < end ≤ original_dim, step > 0
                if (s.start < 0 || s.start >= a->dims[dim]) {
                    throw std::invalid_argument(
                        "get_slice: slice.start (" + std::to_string(s.start)
                        + ") out of range for dim " + std::to_string(dim)
                        + " (size=" + std::to_string(a->dims[dim]) + ")"
                    );
                }
                if (s.end <= s.start || s.end > a->dims[dim]) {
                    throw std::invalid_argument(
                        "get_slice: slice.end (" + std::to_string(s.end)
                        + ") must satisfy start < end ≤ dim size ("
                        + std::to_string(a->dims[dim]) + ")."
                    );
                }
                if (s.step <= 0) {
                    throw std::invalid_argument(
                        "get_slice: slice.step (" + std::to_string(s.step)
                        + ") must be > 0."
                    );
                }
                infos[dim].is_index = false;
                infos[dim].start    = s.start;
                infos[dim].end      = s.end;
                infos[dim].step     = s.step;
            }
        }

        std::vector<int64_t> new_dims;
        std::vector<int64_t> new_strides;
        for (size_t dim = 0; dim < rank; ++dim) {
            if (!infos[dim].is_index) {
                // compute the output length along this axis
                int64_t span = infos[dim].end - infos[dim].start;
                int64_t len  = (span + infos[dim].step - 1) / infos[dim].step;
                new_dims.push_back(len);

                // compute the new stride = old_stride * step
                int64_t orig_stride = a->strides[dim];
                int64_t step        = infos[dim].step;
                new_strides.push_back(orig_stride * step);
            }
        }

        int64_t total_out_elems = 1;
        for (auto d : new_dims) total_out_elems *= d;

        int64_t base_offset_in_elems = 0;
        for (size_t dim = 0; dim < rank; ++dim) {
            if (infos[dim].is_index) {
                base_offset_in_elems += infos[dim].index * a->strides[dim];
            } else {
                base_offset_in_elems += infos[dim].start * a->strides[dim];
            }
        }

        T* orig_raw = reinterpret_cast<T*>( a->data.get() );
        T* sliced_raw = orig_raw + base_offset_in_elems;
        std::shared_ptr<void> alias_data(a->data, static_cast<void*>(sliced_raw));

        a->dims    = std::move(new_dims);
        a->strides = std::move(new_strides);
        a->data    = std::move(alias_data);

        return a;
    }

    // Explicit template instantiations
    #define INSTANTIATE_MEMORY_OPS(T) \
        template std::shared_ptr<DeviceTensor<T>> expand<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t, int64_t); \
        template std::shared_ptr<DeviceTensor<T>> squeeze<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t); \
        template std::shared_ptr<DeviceTensor<T>> unsqueeze<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t); \
        template std::shared_ptr<DeviceTensor<T>> get_slice<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::vector<SliceArg>&);

    INSTANTIATE_MEMORY_OPS(int32_t)
    INSTANTIATE_MEMORY_OPS(int64_t)
    INSTANTIATE_MEMORY_OPS(double)

} // namespace lattica_hw_api
