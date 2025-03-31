
#include "device_memory_impl.h"
#include "g_decomposition.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <omp.h>

namespace lattica_hw_api {
    template <typename T>
    void g_decomposition(
        const std::shared_ptr<DeviceTensor<T>>& a,      // [...], arbitrary shape
        std::shared_ptr<DeviceTensor<T>>& result,       // [..., power] (output)
        size_t power,                                   // Number of digits
        size_t base_bits                                // Base bits (i.e. log₂ base)
    ) {
        const size_t base = 1ULL << base_bits;

        // Validate dimensions
        const auto& in_shape = a->dims;
        const auto& out_shape = result->dims;

        if (out_shape.size() != in_shape.size() + 1 || out_shape.back() != static_cast<int64_t>(power) ||
            !std::equal(in_shape.begin(), in_shape.end(), out_shape.begin())) {
            throw std::invalid_argument("Output must have shape a.shape + [power]");
        }

       // Compute total input elements
        int64_t total = 1;
        for (auto d : in_shape) total *= d;

        // Compute strides for index mapping
        std::vector<int64_t> strides(in_shape.size(), 1);
        for (int i = in_shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * in_shape[i + 1];
        }

        #pragma omp parallel for
        for (int64_t flat_idx = 0; flat_idx < total; ++flat_idx) {
            std::vector<int64_t> coord(in_shape.size());
            int64_t remaining = flat_idx;

            for (size_t i = 0; i < in_shape.size(); ++i) {
                coord[i] = remaining / strides[i];
                remaining %= strides[i];
            }

            T value = a->at(coord);
            std::vector<int64_t> out_coord = coord;
            out_coord.push_back(0);

            for (size_t d = 0; d < power; ++d) {
                out_coord.back() = d;
                result->at(out_coord) = value % base;
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

    template void g_decomposition<int32_t>(
        const std::shared_ptr<DeviceTensor<int32_t>>& a,
        std::shared_ptr<DeviceTensor<int32_t>>& result,
        size_t power,
        size_t base_bits
    );
    template void g_decomposition<int64_t>(
        const std::shared_ptr<DeviceTensor<int64_t>>& a,
        std::shared_ptr<DeviceTensor<int64_t>>& result,
        size_t power,
        size_t base_bits
    );

} // namespace lattica_hw_api

