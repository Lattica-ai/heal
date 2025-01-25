#ifndef PERMUTE_H
#define PERMUTE_H

/**
 * @file permute.h
 * @brief Applies permutations to elements of a vector.
 *
 * This module takes a vector `a` with dimensions `[l, m, k]`,
 * a set of permutations `perms` with dimensions `[l, m]`,
 * and produces a result vector where the permutations are applied
 * element-wise across the first two dimensions.
 *
 * Expected Input Sizes:
 * - Input vector `a` must have dimensions `[l, m, k]`.
 * - Permutations `perms` must have dimensions `[l, m]`.
 * - Result vector `result` must have dimensions `[l, m, k]` (output).
 */

namespace lattica_hw_api {

    template <typename T>
    void permute(
        const std::shared_ptr<DeviceMemory<T>>& a,         // [l, m, k]
        const std::shared_ptr<DeviceMemory<T>>& perms,    // [l, m]
        std::shared_ptr<DeviceMemory<T>>& result          // [l, m, k] (output)
    );

}

#endif // PERMUTE_H
