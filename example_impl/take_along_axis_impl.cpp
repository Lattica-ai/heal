#include "device_memory_impl.h"
#include "take_along_axis.h"
#include <stdexcept>
namespace lattica_hw_api {

template <typename T>
void take_along_axis(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<int64_t>>& indices,
    int64_t axis,
    std::shared_ptr<DeviceTensor<T>>& result)
{
    const int64_t rank = a->dims.size();

    // Normalize & validate axis
    if ((axis < -rank) || (axis >= rank))
        throw std::out_of_range("Axis out of range");
    if (axis < 0) axis += rank;

    // Rank‐match check
    if (indices->dims.size() != rank)
        throw std::invalid_argument("`indices` rank must match `a` rank");

    if (result->dims != a->dims) {
        throw std::invalid_argument("`result` must have identical shape to `a` "
                                    "(this specialised take_along_axis assumes it).");
    }

    /* ---------- strides & total elements ------------------------------ */
    std::vector<int64_t> strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * a->dims[i + 1];

    int64_t total = 1;
    for (auto d : a->dims) total *= d;

    /* ---------- flat parallel loop ------------------------------------ */
    #pragma omp parallel for
    for (int64_t flat = 0; flat < total; ++flat) {
        std::vector<int64_t> coord(rank);
        int64_t rem = flat;
        for (size_t i = 0; i < rank; ++i) {
            coord[i] = rem / strides[i];
            rem %= strides[i];
        }

        /* gather index with broadcasting */
        int64_t sel = indices->at_with_broadcast(coord);

        const int64_t axis_size = a->dims[axis];
        if (sel < 0) sel += axis_size;
        if (sel < 0 || sel >= axis_size)
            throw std::out_of_range("Index out of bounds in take_along_axis");

        /* build source coord & scatter */
        std::vector<int64_t> src_coord = coord;
        src_coord[axis] = sel;
        result->at(coord) = a->at(src_coord);
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
