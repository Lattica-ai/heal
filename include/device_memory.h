#pragma once

#include "device_tensor_ex_impl.h"
#include <torch/torch.h>

namespace lattica_hw_api {

/**
 * @brief Allocate a new device tensor on hardware with all elements initialized to zero.
 * @param dims Shape of the tensor.
 */
template <typename T>
std::shared_ptr<DeviceTensor<T>> zeros(const std::vector<int64_t>& dims);

/**
 * @brief Allocate a new device tensor on hardware without initializing elements.
 * @param dims Shape of the tensor.
 */
template <typename T>
std::shared_ptr<DeviceTensor<T>> empty(const std::vector<int64_t>& dims);

/**
 * @brief Upload a PyTorch tensor to device memory.
 * @param tensor A contiguous torch::Tensor of type T.
 */
template <typename T>
std::shared_ptr<DeviceTensor<T>> host_to_device(const torch::Tensor& tensor);

/**
 * @brief Download a device tensor back into a torch::Tensor.
 * @param memory A shared pointer to the device tensor.
 */
template <typename T>
torch::Tensor device_to_host(const std::shared_ptr<DeviceTensor<T>>& memory);

} // namespace lattica_hw_api
