#pragma once

#include "device_tensor.h"
#include <memory>

namespace lattica_hw_api {

template <typename T>
void apply_g_decomp_relative_to_full_q(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& q_list,
    const std::shared_ptr<DeviceTensor<T>>& q_inv,
    int g_exp,
    int g_base_bits,
    int level_size_bits,
    const std::shared_ptr<DeviceTensor<T>>& level_inv,
    std::shared_ptr<DeviceTensor<T>>& out
);

} // namespace lattica_hw_api