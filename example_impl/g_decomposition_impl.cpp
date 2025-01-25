
#include "device_memory_impl.h"
#include "g_decomposition.h"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace lattica_hw_api {

template <typename T>
void g_decomposition(
    const std::shared_ptr<DeviceMemory<T>>& a,      // [m]
    std::shared_ptr<DeviceMemory<T>>& result,       // [m, power] (output)
    size_t power,                                // Number of digits
    size_t base_bits                             // Base bits
) {
    if (result->dims[0] != a->dims[0] || result->dims[1] != power) {
        throw std::invalid_argument("Dimensions of input vector and result vector do not match.");
    }

    size_t base = 1 << base_bits; // Calculate base as 2^base_bits

    for (size_t i = 0; i < a->dims[0]; ++i) {
        T value = a->at({i});
        for (size_t j = 0; j < power; ++j) {
            result->at({i, j}) = value % base;
            value /= base;
        }
        if (value > 0) {
            std::cerr << "Warning: value at index " << i << " exceeds representation capacity with given power and base_bits.\n";
        }
    }
}
template void g_decomposition<int32_t>(
    const std::shared_ptr<DeviceMemory<int32_t>>& a,
    std::shared_ptr<DeviceMemory<int32_t>>& result,
    size_t power,
    size_t base_bits
);
template void g_decomposition<int64_t>(
    const std::shared_ptr<DeviceMemory<int64_t>>& a,
    std::shared_ptr<DeviceMemory<int64_t>>& result,
    size_t power,
    size_t base_bits
);

} // namespace lattica_hw_api

