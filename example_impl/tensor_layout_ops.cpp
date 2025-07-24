#include "tensor_layout_ops.h"
#include "device_tensor_ex.h"
#include <stdexcept>
#include <functional>

namespace lattica_hw_api {

template <typename T>
std::shared_ptr<DeviceTensor<T>> expand(
    const std::shared_ptr<DeviceTensor<T>>& a,
    int64_t axis,
    int64_t repeats)
{
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
    return std::make_shared<DeviceTensor<T>>(new_dims, new_strides, a->data);
}

template <typename T>
std::shared_ptr<DeviceTensor<T>> squeeze(
    const std::shared_ptr<DeviceTensor<T>>& a,
    int64_t axis)
{
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

    return std::make_shared<DeviceTensor<T>>(new_dims, new_strides, a->data);
}

template <typename T>
std::shared_ptr<DeviceTensor<T>> unsqueeze(
    const std::shared_ptr<DeviceTensor<T>>& a,
    int64_t axis)
{
    int64_t ndim = static_cast<int64_t>(a->dims.size());
    if (axis < 0) axis += (ndim + 1);
    if (axis < 0 || axis > ndim) {
        throw std::invalid_argument("Invalid unsqueeze dimension.");
    }

    std::vector<int64_t> new_dims = a->dims;
    std::vector<int64_t> new_strides = a->strides;
    new_dims.insert(new_dims.begin() + axis, 1);
    new_strides.insert(new_strides.begin() + axis, 0);

    return std::make_shared<DeviceTensor<T>>(new_dims, new_strides, a->data);
}


template <typename T>
std::shared_ptr<DeviceTensor<T>> moveaxis(
    const std::shared_ptr<DeviceTensor<T>>& a,
    int64_t  axis_src,
    int64_t  axis_dst)
{
    // ── 0. Basic sanity checks ───────────────────────────────────────────────
    if (!a) {
        throw std::invalid_argument("moveaxis: tensor pointer is null.");
    }

    const int64_t ndim = static_cast<int64_t>(a->dims.size());

    // Normalise negative indices
    if (axis_src < 0) axis_src += ndim;
    if (axis_dst < 0) axis_dst += ndim;

    // Validate indices
    if (axis_src < 0 || axis_src >= ndim ||
        axis_dst < 0 || axis_dst >= ndim) {
        throw std::invalid_argument("moveaxis: axis index out of range.");
    }

    // No-op fast-path
    if (axis_src == axis_dst) {
        return a;
    }

    // ── 1. Re-order dims & strides ───────────────────────────────────────────
    std::vector<int64_t> new_dims = a->dims;
    std::vector<int64_t> new_strides = a->strides;

    const int64_t dim_val = new_dims[axis_src];
    const int64_t stride_val = new_strides[axis_src];

    // Remove source position first
    new_dims.erase(new_dims.begin() + axis_src);
    new_strides.erase(new_strides.begin() + axis_src);

    // Insert the axis metadata in its new place
    new_dims.insert(new_dims.begin() + axis_dst, dim_val);
    new_strides.insert(new_strides.begin() + axis_dst, stride_val);

    // ── 2. Commit the modified metadata ──────────────────────────────────────
    return std::make_shared<DeviceTensor<T>>(new_dims, new_strides, a->data);
}

template<typename T>
std::shared_ptr<DeviceTensor<T>> get_slice(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::vector<SliceArg>& slices)
{
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

    int64_t base_offset_in_elems = 0;
    for (size_t dim = 0; dim < rank; ++dim) {
        if (infos[dim].is_index) {
            base_offset_in_elems += infos[dim].index * a->strides[dim];
        } else {
            base_offset_in_elems += infos[dim].start * a->strides[dim];
        }
    }

    T* orig_raw = reinterpret_cast<T*>(a->data.get());
    T* view_raw = orig_raw + base_offset_in_elems;
    std::shared_ptr<void> alias_data(
        a->data,                  // share refcount & deleter
        static_cast<void*>(view_raw) // new pointer into that buffer
    );

    return std::make_shared<DeviceTensor<T>>(
        std::move(new_dims),
        std::move(new_strides),
        std::move(alias_data)
    );
}

template <typename T>
std::shared_ptr<DeviceTensor<T>> flatten(
    const std::shared_ptr<DeviceTensor<T>>& a,
    int64_t start_axis,
    int64_t end_axis)
{
    if (!a->is_contiguous()) {
        throw std::runtime_error("flatten: input tensor must be contiguous");
    }

    const auto& dims = a->dims;
    const auto& strides = a->strides;
    int64_t ndim = static_cast<int64_t>(dims.size());

    // Wrap negatives
    if (start_axis < 0) start_axis += ndim;
    if (end_axis < 0) end_axis += ndim;

    // Validate
    if (start_axis < 0 || start_axis >= ndim || end_axis < start_axis || end_axis >= ndim) {
        throw std::invalid_argument("flatten: invalid start_axis/end_axis");
    }

    // Compute size of the flattened dimension and its stride
    int64_t flat_size = 1;
    int64_t flat_stride = strides[end_axis];
    for (int64_t i = start_axis; i <= end_axis; ++i) {
        flat_size *= dims[i];
    }

    // Build new dims/strides
    std::vector<int64_t> new_dims;
    std::vector<int64_t> new_strides;
    new_dims.reserve(ndim - (end_axis - start_axis));
    new_strides.reserve(ndim - (end_axis - start_axis));

    // Copy dims/strides before start_axis
    new_dims.insert(new_dims.end(), dims.begin(), dims.begin() + start_axis);
    new_strides.insert(new_strides.end(), strides.begin(), strides.begin() + start_axis);

    // Insert flattened dim
    new_dims.push_back(flat_size);
    new_strides.push_back(flat_stride);

    // Copy dims/strides after end_axis
    new_dims.insert(new_dims.end(), dims.begin() + end_axis + 1, dims.end());
    new_strides.insert(new_strides.end(), strides.begin() + end_axis + 1, strides.end());

    // Return a new tensor view (sharing the same data)
    return std::make_shared<DeviceTensor<T>>(new_dims, new_strides, a->data);
}


template <typename T>
std::shared_ptr<DeviceTensor<T>> reshape(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::vector<int64_t>& new_dims)
{
    int64_t new_total = 1;
    for (int64_t d : new_dims) new_total *= d;

    int64_t current_total = a->numel();

    if (new_total != current_total) {
        throw std::invalid_argument("Total size of new shape must match number of elements (excluding broadcasted dims).");
    }

    // Generate new strides
    std::vector<int64_t> new_strides(new_dims.size());
    int64_t stride = 1;
    for (int64_t i = new_dims.size() - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= new_dims[i];
    }

    // If this is a broadcasted tensor (has zero strides), keep them zero in broadcasted dimensions
    // and otherwise use normal C-contiguous layout
    bool has_broadcast = std::any_of(a->strides.begin(), a->strides.end(), [](int64_t s) { return s == 0; });

    if (has_broadcast) {
        // Fallback: zero all strides if any broadcasting involved
        // More precise reuse of original strides would require complex mapping
        for (int64_t i = 0; i < static_cast<int64_t>(new_strides.size()); ++i) {
            if (new_dims[i] != 1) {
                new_strides[i] = 1;
                for (int64_t j = i + 1; j < static_cast<int64_t>(new_dims.size()); ++j) {
                    new_strides[i] *= new_dims[j];
                }
                break; // Keep only one base dimension
            }
        }
    }

    return std::make_shared<DeviceTensor<T>>(new_dims, new_strides, a->data);
}

// Explicit template instantiations
#define FUNCTIONS_WITH_INT8_INSTANCIATION(T) \
    template std::shared_ptr<DeviceTensor<T>> moveaxis<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t, int64_t); \
    template std::shared_ptr<DeviceTensor<T>> expand<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t, int64_t); \
    template std::shared_ptr<DeviceTensor<T>> reshape<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::vector<int64_t>&); \
    template std::shared_ptr<DeviceTensor<T>> get_slice<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::vector<SliceArg>&);

#define INSTANTIATE_ALL_FUNCTIONS(T) \
    FUNCTIONS_WITH_INT8_INSTANCIATION(T) \
    template std::shared_ptr<DeviceTensor<T>> squeeze<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t); \
    template std::shared_ptr<DeviceTensor<T>> unsqueeze<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t); \
    template std::shared_ptr<DeviceTensor<T>> flatten<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t, int64_t);

// Instantiate all memory operations for int32_t and int64_t.
INSTANTIATE_ALL_FUNCTIONS(int32_t)
INSTANTIATE_ALL_FUNCTIONS(int64_t)

// Instantiate functions thet needs int8_t
FUNCTIONS_WITH_INT8_INSTANCIATION(int8_t)

} // namespace lattica_hw_api
