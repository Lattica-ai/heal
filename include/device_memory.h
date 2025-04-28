#ifndef DeviceTensor_H
#define DeviceTensor_H

#include <vector>
#include <memory>
#include <torch/torch.h>

/**
 * @brief Abstract device-side tensor for hardware-accelerated memory.
 */
template <typename T>
struct DeviceTensor {
    void reshape(const std::vector<int64_t>& new_dims);
    void print() const;
};

namespace lattica_hw_api {

/**
 * @brief Allocate a new uninitialized device tensor on hardware.
 * @param dims Shape of the tensor.
 * @return A shared pointer to the newly allocated tensor.
 */
template <typename T>
std::shared_ptr<DeviceTensor<T>> empty(const std::vector<int64_t>& dims);

/**
 * @brief Allocate a new device tensor on hardware and initialize all elements to zero.
 * @param dims Shape of the tensor.
 * @return A shared pointer to the newly allocated tensor with zeros.
 */
template <typename T>
std::shared_ptr<DeviceTensor<T>> zeros(const std::vector<int64_t>& dims);

/**
 * @brief Upload a PyTorch tensor to device memory.
 * @param tensor A contiguous torch::Tensor of type T.
 * @return A shared pointer to the device tensor containing the uploaded data.
 */
template <typename T>
std::shared_ptr<DeviceTensor<T>> host_to_device(const torch::Tensor& tensor);

/**
 * @brief Download a device tensor back into a torch::Tensor.
 * @param memory A shared pointer to the device tensor.
 * @return A torch::Tensor containing a copy of the device tensor data.
 */
template <typename T>
torch::Tensor device_to_host(const std::shared_ptr<DeviceTensor<T>>& memory);

} // namespace lattica_hw_api

#endif // DeviceTensor_H
