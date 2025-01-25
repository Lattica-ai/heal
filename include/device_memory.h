#ifndef DEVICEMEMORY_H
#define DEVICEMEMORY_H

#include <vector>
#include <memory>

#include "nested_vector.h"

/**
 * @brief A structure to manage up to 4D vectors for hardware memory.
 * @tparam T The data type of the vector elements.
 */
template <typename T>
struct DeviceMemory {
    void print();
    void reshape(const std::vector<size_t>& new_dims);
};

namespace lattica_hw_api {

    /**
     * Allocates hardware memory for a vector of up to 4D.
     * @tparam T The data type of the vector elements.
     * @param dims Vector of dimensions (size <= 4).
     * @return A shared pointer to the allocated DeviceMemory.
     */
    template <typename T>
    std::shared_ptr<DeviceMemory<T>> allocate_on_hardware(const std::vector<size_t>& dims);

    /**
     * Move data from the CPU to hardware memory.
     * Supports 1D to 4D vectors.
     * @tparam T The data type of the vector elements.
     * @tparam Dim The dimension of the vector (1D to 4D).
     * @param cpu_data A nested vector representing the vector data.
     * @return A shared pointer to the allocated DeviceMemory.
     */
    template <typename T, size_t Dim>
    std::shared_ptr<DeviceMemory<T>> move_to_hardware(const NestedVectorType<T, Dim>& cpu_data);

    /**
     * Moves a vector from hardware memory back to CPU.
     * @tparam T The data type of the vector elements.
     * @tparam N The dimension of the vector (1D to 4D).
     * @param hw_memory A shared pointer to the DeviceMemory.
     * @return A vector containing the data.
     */
    template <typename T, size_t N>
    NestedVectorType<T, N> move_from_hardware(const std::shared_ptr<DeviceMemory<T>>& hw_memory);

}

#endif // DEVICEMEMORY_H
