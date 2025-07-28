
#include "gadget_decomposition.h"
#include "device_tensor_ex.h"
#include <stdexcept>
#include <iostream>

namespace lattica_hw_api {

template <typename T, typename U>
void apply_g_decomp(
    const std::shared_ptr<DeviceTensor<T>>& a,      // [...], arbitrary shape
    size_t power,                                   // Number of digits
    size_t base_bits,                               // Base bits (i.e. log₂ base)
    std::shared_ptr<DeviceTensor<U>>& result)       // [..., power] (output)
{
    const size_t base = 1ULL << base_bits;

    // Validate dimensions
    const auto& in_shape = a->dims;
    const auto& out_shape = result->dims;

    if (out_shape.size() != in_shape.size() + 1 || out_shape.back() != static_cast<int64_t>(power) ||
        !std::equal(in_shape.begin(), in_shape.end(), out_shape.begin())) {
        throw std::invalid_argument("Output must have shape a.shape + [power]");
    }

    int64_t total = a->numel();
    std::vector<int64_t> strides = DeviceTensor<T>::compute_contiguous_strides(in_shape);

    #pragma omp parallel for
    for (int64_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<int64_t> coord = DeviceTensor<T>::unravel_index(flat_idx, in_shape, strides);
        T value = a->at(coord);
        std::vector<int64_t> out_coord = coord;
        out_coord.push_back(0);

        for (size_t d = 0; d < power; ++d) {
            out_coord.back() = d;
            result->at(out_coord) = static_cast<U>(value % base);
            value /= base;
        }

        if (value > 0) {
            #pragma omp critical
            {
                std::cerr << "Warning: value at ";
                for (auto x : coord) std::cerr << x << " ";
                std::cerr << "exceeds capacity with base_bits=" << base_bits << " and power=" << power << "\n";
            }
        }
    }
}


#define INSTANTIATE_APPLY_G_DECOMP(T1, T2) \
    template void apply_g_decomp<T1, T2>( \
        const std::shared_ptr<DeviceTensor<T1>>& a, \
        size_t power, \
        size_t base_bits, \
        std::shared_ptr<DeviceTensor<T2>>& result \
    );

INSTANTIATE_APPLY_G_DECOMP(int32_t, int8_t)
INSTANTIATE_APPLY_G_DECOMP(int64_t, int8_t)
INSTANTIATE_APPLY_G_DECOMP(int32_t, int32_t)
INSTANTIATE_APPLY_G_DECOMP(int64_t, int64_t)

} // namespace lattica_hw_api

