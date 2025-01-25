#include <cstdint>
#include <type_traits>

template <typename T>
struct TypeMapper;

// Specialize the mapper for specific types
template <>
struct TypeMapper<int32_t> {
    using type = int64_t;
};

template <>
struct TypeMapper<int64_t> {
    using type = __int128_t; // Use compiler-specific 128-bit integer type
};

// Helper alias for convenience
template <typename T>
using T_DP = typename TypeMapper<T>::type;