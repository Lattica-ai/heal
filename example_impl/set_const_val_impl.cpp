#include "device_memory_impl.h"
#include "set_const_val.h"
#include <stdexcept>

namespace lattica_hw_api {

template <typename T>
std::shared_ptr<DeviceTensor<T>> set_const_val(
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

    return a;
}


// explicit instantiation for set_const_val
template std::shared_ptr<DeviceTensor<int32_t>> set_const_val<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&, int32_t);
template std::shared_ptr<DeviceTensor<int64_t>> set_const_val<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&, int64_t);

} // namespace lattica_hw_api