#include "device_memory_impl.h"
#include <cstring>
#include <stdexcept>
#include <numeric>

namespace lattica_hw_api {

    /**
     * Allocate hardware memory for a vector with the specified dimensions.
     * Supports vectors of 1D to 4D by setting unused dimensions to 1.
     * @tparam T The data type of the vector elements.
     * @param dims A vector of dimensions.
     * @return A shared pointer to the allocated DeviceMemory.
     */
    template <typename T>
    std::shared_ptr<DeviceMemory<T>> allocate_on_hardware(const std::vector<size_t>& dims) {
        return std::make_shared<DeviceMemory<T>>(dims);
    }

    // helper function for move_to_hardware
    template <typename T, size_t Dim>
    void _allocate_and_copy(const NestedVectorType<T, Dim>& data, std::shared_ptr<DeviceMemory<T>>& hw_memory, size_t offset, const std::vector<size_t>& dims) {
        if constexpr (Dim == 1) {
            // Handle 1D vector
            std::memcpy(hw_memory->data.get() + offset, data.data(), dims.back() * sizeof(T));
        } else {
            // Handle nested vectors
            size_t stride = 1;
            for (size_t i = dims.size() - Dim + 1; i < dims.size(); ++i) {
                stride *= dims[i];
            }
            for (size_t i = 0; i < data.size(); ++i) {
                _allocate_and_copy<T, Dim - 1>(data[i], hw_memory, offset + i * stride, dims);
            }
        }
    }

    // Explicit template instantiations for Dim = 1 to 4
    template void _allocate_and_copy<int, 1>(const NestedVectorType<int, 1>&, std::shared_ptr<DeviceMemory<int>>&, size_t, const std::vector<size_t>&);
    template void _allocate_and_copy<int, 2>(const NestedVectorType<int, 2>&, std::shared_ptr<DeviceMemory<int>>&, size_t, const std::vector<size_t>&);
    template void _allocate_and_copy<int, 3>(const NestedVectorType<int, 3>&, std::shared_ptr<DeviceMemory<int>>&, size_t, const std::vector<size_t>&);
    template void _allocate_and_copy<int, 4>(const NestedVectorType<int, 4>&, std::shared_ptr<DeviceMemory<int>>&, size_t, const std::vector<size_t>&);


    /**
     * Move data from the CPU to hardware memory.
     * Automatically infers dimensions based on the input vector's structure.
     * Supports 1D to 4D vectors.
     * @tparam T The data type of the vector elements.
     * @tparam Dim The dimension of the vector (1D to 4D).
     * @param cpu_data A nested vector representing the vector data.
     * @return A shared pointer to the allocated DeviceMemory.
     */
    template <typename T, size_t Dim>
    std::shared_ptr<DeviceMemory<T>> move_to_hardware(const NestedVectorType<T, Dim>& cpu_data) {
        static_assert(Dim >= 1 && Dim <= 4, "Dim must be between 1 and 4.");

        std::vector<size_t> dims(Dim);
        dims[0] = cpu_data.size();
        if constexpr (Dim >= 2) dims[1] = cpu_data[0].size();
        if constexpr (Dim >= 3) dims[2] = cpu_data[0][0].size();
        if constexpr (Dim >= 4) dims[3] = cpu_data[0][0][0].size();

        auto hw_memory = allocate_on_hardware<T>(dims);

        _allocate_and_copy<T, Dim>(cpu_data, hw_memory, 0, dims);
        return hw_memory;
    }
    template std::shared_ptr<DeviceMemory<int32_t>> move_to_hardware<int32_t, 1>(const NestedVectorType<int32_t, 1>&);
    template std::shared_ptr<DeviceMemory<int32_t>> move_to_hardware<int32_t, 2>(const NestedVectorType<int32_t, 2>&);
    template std::shared_ptr<DeviceMemory<int32_t>> move_to_hardware<int32_t, 3>(const NestedVectorType<int32_t, 3>&);
    template std::shared_ptr<DeviceMemory<int32_t>> move_to_hardware<int32_t, 4>(const NestedVectorType<int32_t, 4>&);
    template std::shared_ptr<DeviceMemory<int64_t>> move_to_hardware<int64_t, 1>(const NestedVectorType<int64_t, 1>&);
    template std::shared_ptr<DeviceMemory<int64_t>> move_to_hardware<int64_t, 2>(const NestedVectorType<int64_t, 2>&);
    template std::shared_ptr<DeviceMemory<int64_t>> move_to_hardware<int64_t, 3>(const NestedVectorType<int64_t, 3>&);
    template std::shared_ptr<DeviceMemory<int64_t>> move_to_hardware<int64_t, 4>(const NestedVectorType<int64_t, 4>&);


    /**
     * Move data from hardware memory back to the CPU.
     * Automatically creates the appropriate nested vector structure.
     * Supports 1D to 4D vectors.
     * @tparam T The data type of the vector elements.
     * @tparam Dim The dimension of the vector (1D to 4D).
     * @param hw_memory A shared pointer to the DeviceMemory.
     * @return A nested vector representing the vector data.
     */
    template <typename T, size_t Dim>
    NestedVectorType<T, Dim> move_from_hardware(const std::shared_ptr<DeviceMemory<T>>& hw_memory) {
        static_assert(Dim >= 1 && Dim <= 4, "Dim must be between 1 and 4.");


        using TensorType = typename NestedVector<T, Dim>::type;

        // Calculate dimensions
        size_t sizes[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < Dim; ++i) {
            sizes[i] = (i < hw_memory->dims.size()) ? hw_memory->dims[i] : 1;
        }

        // Initialize the result vector
        TensorType result;

        // Compile-time dimension specialization
        if constexpr (Dim == 1) {
            result.resize(sizes[0]);
            std::memcpy(result.data(), hw_memory->data.get(), sizes[0] * sizeof(T));
        } else if constexpr (Dim == 2) {
            result.resize(sizes[0]);
            for (size_t i = 0; i < sizes[0]; ++i) {
                result[i].resize(sizes[1]);
                std::memcpy(result[i].data(), hw_memory->data.get() + (i * sizes[1]), sizes[1] * sizeof(T));
            }
        } else if constexpr (Dim == 3) {
            result.resize(sizes[0]);
            for (size_t i = 0; i < sizes[0]; ++i) {
                result[i].resize(sizes[1]);
                for (size_t j = 0; j < sizes[1]; ++j) {
                    result[i][j].resize(sizes[2]);
                    std::memcpy(result[i][j].data(),
                                hw_memory->data.get() + (i * sizes[1] * sizes[2]) + (j * sizes[2]),
                                sizes[2] * sizeof(T));
                }
            }
        } else if constexpr (Dim == 4) {
            result.resize(sizes[0]);
            for (size_t i = 0; i < sizes[0]; ++i) {
                result[i].resize(sizes[1]);
                for (size_t j = 0; j < sizes[1]; ++j) {
                    result[i][j].resize(sizes[2]);
                    for (size_t k = 0; k < sizes[2]; ++k) {
                        result[i][j][k].resize(sizes[3]);
                        std::memcpy(result[i][j][k].data(),
                                    hw_memory->data.get() +
                                        (i * sizes[1] * sizes[2] * sizes[3]) +
                                        (j * sizes[2] * sizes[3]) +
                                        (k * sizes[3]),
                                    sizes[3] * sizeof(T));
                    }
                }
            }
        }


        return result;
    }
    template NestedVectorType<int32_t, 1> move_from_hardware<int32_t, 1>(const std::shared_ptr<DeviceMemory<int32_t>>& hw_memory);
    template NestedVectorType<int32_t, 2> move_from_hardware<int32_t, 2>(const std::shared_ptr<DeviceMemory<int32_t>>& hw_memory);
    template NestedVectorType<int32_t, 3> move_from_hardware<int32_t, 3>(const std::shared_ptr<DeviceMemory<int32_t>>& hw_memory);
    template NestedVectorType<int32_t, 4> move_from_hardware<int32_t, 4>(const std::shared_ptr<DeviceMemory<int32_t>>& hw_memory);
    template NestedVectorType<int64_t, 1> move_from_hardware<int64_t, 1>(const std::shared_ptr<DeviceMemory<int64_t>>& hw_memory);
    template NestedVectorType<int64_t, 2> move_from_hardware<int64_t, 2>(const std::shared_ptr<DeviceMemory<int64_t>>& hw_memory);
    template NestedVectorType<int64_t, 3> move_from_hardware<int64_t, 3>(const std::shared_ptr<DeviceMemory<int64_t>>& hw_memory);
    template NestedVectorType<int64_t, 4> move_from_hardware<int64_t, 4>(const std::shared_ptr<DeviceMemory<int64_t>>& hw_memory);

} // namespace lattica_hw_api

// Recursive helper function to print braces
template <typename T>
void _print_device_memory_recursive(DeviceMemory<T>* vector,
                                   const std::vector<size_t>& dims,
                                   std::vector<size_t> indices,
                                   size_t current_dim) {
    if (current_dim == dims.size()) {
        // Base case: print the value at the current indices
        std::cout << vector->at(indices);
        return;
    }

    // Recursive case: iterate over the current dimension
    std::cout << "{";
    for (size_t i = 0; i < dims[current_dim]; ++i) {
        indices[current_dim] = i;
        if (i > 0) std::cout << ", ";
        _print_device_memory_recursive(vector, dims, indices, current_dim + 1);
    }
    std::cout << "}";
}

// Wrapper function to print the vector shape and hierarchy
template <typename T>
void DeviceMemory<T>::print() {
    // Print the active shape of the vector
    std::cout << "\nTensor Shape: [";
    for (size_t i = 0; i < this->dims.size(); ++i) {
        std::cout << this->dims[i];
        if (i < this->dims.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Print the vector hierarchy
    std::cout << "Tensor Data: ";
    _print_device_memory_recursive(this, this->dims, std::vector<size_t>(this->dims.size(), 0), 0);
    std::cout << "\n\n";
}
template void DeviceMemory<int32_t>::print();
template void DeviceMemory<int64_t>::print();

/**
 * Reshape the vector to new dimensions.
 * @param new_dims Vector of new dimensions (size <= 4).
 * Throws std::invalid_argument if the total size does not match.
 */
template <typename T>
void DeviceMemory<T>::reshape(const std::vector<size_t>& new_dims) {
    if (new_dims.size() > 4) {
        throw std::invalid_argument("DeviceMemory supports up to 4 dimensions.");
    }
    size_t new_total_size = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<size_t>());
    size_t current_total_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    if (new_total_size != current_total_size) {
        throw std::invalid_argument("Total size of new dimensions must match the current size.");
    }
    dimensions = new_dims.size();
    dims = new_dims;
}
template void DeviceMemory<int32_t>::reshape(const std::vector<size_t>&);
template void DeviceMemory<int64_t>::reshape(const std::vector<size_t>&);
