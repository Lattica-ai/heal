
#include "device_memory_impl.h"
#include "modsum.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>

#include "typing.h"

namespace lattica_hw_api {

template <typename T>
void modsum_v1(
    const std::shared_ptr<DeviceMemory<T>>& a,  // [m, k]
    const std::shared_ptr<DeviceMemory<T>>& b,  // [m, k]
    const std::shared_ptr<DeviceMemory<T>>& p,  //    [k]
    std::shared_ptr<DeviceMemory<T>>& result    // [m, k] (output)
) {
    if (a->dims != b->dims || a->dims != result->dims || a->dims[1] != p->dims[0]) {
        throw std::invalid_argument("Dimensions of input vectors do not match.");
    }

    for (size_t i = 0; i < a->dims[0]; ++i) {
        for (size_t j = 0; j < a->dims[1]; ++j) {
            T_DP<T> temp = static_cast<T_DP<T>>(a->at({i, j})) + static_cast<T_DP<T>>(b->at({i, j}));
            result->at({i, j}) = static_cast<T>(temp % static_cast<T_DP<T>>(p->at({j})));
        }
    }
}
template void modsum_v1<int32_t>(
    const std::shared_ptr<DeviceMemory<int32_t>>& a,
    const std::shared_ptr<DeviceMemory<int32_t>>& b,
    const std::shared_ptr<DeviceMemory<int32_t>>& p,
    std::shared_ptr<DeviceMemory<int32_t>>& result
);
template void modsum_v1<int64_t>(
    const std::shared_ptr<DeviceMemory<int64_t>>& a,
    const std::shared_ptr<DeviceMemory<int64_t>>& b,
    const std::shared_ptr<DeviceMemory<int64_t>>& p,
    std::shared_ptr<DeviceMemory<int64_t>>& result
);

template <typename T>
void modsum_v2(
    const std::shared_ptr<DeviceMemory<T>>& a,  // [m, k]
    const std::shared_ptr<DeviceMemory<T>>& b,  //    [k]
    const std::shared_ptr<DeviceMemory<T>>& p,  //    [k]
    std::shared_ptr<DeviceMemory<T>>& result    // [m, k] (output)
) {
    if (a->dims[1] != b->dims[0] || a->dims[1] != p->dims[0] || a->dims != result->dims) {
        throw std::invalid_argument("Dimensions of input vectors do not match.");
    }

    for (size_t i = 0; i < a->dims[0]; ++i) {
        for (size_t j = 0; j < a->dims[1]; ++j) {
            T_DP<T> temp = static_cast<T_DP<T>>(a->at({i, j})) + static_cast<T_DP<T>>(b->at({j}));
            result->at({i, j}) = static_cast<T>(temp % static_cast<T_DP<T>>(p->at({j})));
        }
    }
}
template void modsum_v2<int32_t>(
    const std::shared_ptr<DeviceMemory<int32_t>>& a,
    const std::shared_ptr<DeviceMemory<int32_t>>& b,
    const std::shared_ptr<DeviceMemory<int32_t>>& p,
    std::shared_ptr<DeviceMemory<int32_t>>& result
);
template void modsum_v2<int64_t>(
    const std::shared_ptr<DeviceMemory<int64_t>>& a,
    const std::shared_ptr<DeviceMemory<int64_t>>& b,
    const std::shared_ptr<DeviceMemory<int64_t>>& p,
    std::shared_ptr<DeviceMemory<int64_t>>& result
);

template <typename T>
void modsum_v3(
    const std::shared_ptr<DeviceMemory<T>>& a,  // [m, k]
    T b,                              // scalar
    const std::shared_ptr<DeviceMemory<T>>& p,  //    [k]
    std::shared_ptr<DeviceMemory<T>>& result    // [m, k] (output)
) {
    if (a->dims[1] != p->dims[0] || a->dims != result->dims) {
        throw std::invalid_argument("Dimensions of input vectors do not match.");
    }

    for (size_t i = 0; i < a->dims[0]; ++i) {
        for (size_t j = 0; j < a->dims[1]; ++j) {
            T_DP<T> temp = static_cast<T_DP<T>>(a->at({i, j})) + static_cast<T_DP<T>>(b);
            result->at({i, j}) = static_cast<T>(temp % static_cast<T_DP<T>>(p->at({j})));
        }
    }
}
template void modsum_v3<int32_t>(
    const std::shared_ptr<DeviceMemory<int32_t>>& a,
    int32_t b,
    const std::shared_ptr<DeviceMemory<int32_t>>& p,
    std::shared_ptr<DeviceMemory<int32_t>>& result
);
template void modsum_v3<int64_t>(
    const std::shared_ptr<DeviceMemory<int64_t>>& a,
    int64_t b,
    const std::shared_ptr<DeviceMemory<int64_t>>& p,
    std::shared_ptr<DeviceMemory<int64_t>>& result
);

} // namespace lattica_hw_api

