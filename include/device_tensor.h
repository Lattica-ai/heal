#pragma once

/**
 * @brief User-Defined Device Tensor Interface
 *
 * This is a forward declaration of the DeviceTensor class template.
 *
 * To use the lattica_hw_api functions, **you must provide your own implementation**
 * of `DeviceTensor<T>`, which represents a tensor or buffer stored in your device's memory (e.g., GPU, FPGA, ASIC).
 *
 * @tparam T Element type stored in the device tensor (e.g., float, int32_t).
 */

template <typename T>
class DeviceTensor;