#ifndef NESTED_VECTOR_H
#define NESTED_VECTOR_H

#include <iostream>
#include <vector>

// Helper to recursively define nested vector types
template <typename T, size_t N>
struct NestedVector {
    using type = std::vector<typename NestedVector<T, N - 1>::type>;
};

template <typename T>
struct NestedVector<T, 1> {
    using type = std::vector<T>;
};

template <typename T, size_t N>
using NestedVectorType = typename NestedVector<T, N>::type;

// Helper to deduce depth (dimensions)
template <typename T>
struct is_nested_vector {
    static constexpr size_t value = 0;
};

template <typename T>
struct is_nested_vector<std::vector<T>> {
    static constexpr size_t value = 1 + is_nested_vector<T>::value;
};

// Print nested vector
template <typename NestedTensor>
void print_nested_vector(const NestedTensor& vector, size_t current_depth = 0) {
    using ValueType = typename NestedTensor::value_type;

    if constexpr (std::is_arithmetic_v<ValueType>) {
        // Base case: 1D vector
        std::cout << "{";
        for (size_t i = 0; i < vector.size(); ++i) {
            std::cout << vector[i];
            if (i < vector.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}";
    } else {
        // Recursive case: nested vector
        std::cout << "{";
        for (size_t i = 0; i < vector.size(); ++i) {
            print_nested_vector(vector[i], current_depth + 1);
            if (i < vector.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}";
    }

    if (current_depth == 0) {
        std::cout << std::endl; // Final newline at the top-level vector
    }
}

#endif // NESTED_VECTOR_H
