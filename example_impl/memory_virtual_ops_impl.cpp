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

    // Explicit template instantiations
    #define INSTANTIATE_MEMORY_OPS(T) \
        template std::shared_ptr<DeviceTensor<T>> expand<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t, int64_t); \
        template std::shared_ptr<DeviceTensor<T>> squeeze<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t); \
        template std::shared_ptr<DeviceTensor<T>> unsqueeze<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t);

    INSTANTIATE_MEMORY_OPS(int32_t)
    INSTANTIATE_MEMORY_OPS(int64_t)
    INSTANTIATE_MEMORY_OPS(double)

} // namespace lattica_hw_api
