#ifndef COMPUTE_OPS_H
#define COMPUTE_OPS_H

#include <memory>

/**
 * @file compute_ops.h
 * @brief Provides additional compute operations supporting specialized FHE workloads.
 *
 * This module implements functions for bit‐level operations, indexing, scaling, rotation,
 * masking, and other transformation steps required by specific FHE schemes or
 * application‐level optimizations.
 *
 * All functions live in the `lattica_hw_api` namespace and operate on
 * DeviceTensor<T> objects in device memory, supporting broadcasting
 * and non‐contiguous layouts via stride‐aware indexing.
 */

namespace lattica_hw_api {

    using IndexType = int64_t;

    // —————————————————————————————————————————————————————————————————————————

    template <typename T>
    void take_along_axis(
        const std::shared_ptr<DeviceTensor<T>>&           a,
        const std::shared_ptr<DeviceTensor<IndexType>>&   indices,
        IndexType                                         axis,
        std::shared_ptr<DeviceTensor<T>>&                 result
    );


} // namespace lattica_hw_api

#endif // COMPUTE_OPS_H
