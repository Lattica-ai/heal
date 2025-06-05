#include "device_memory_impl.h"
#include "modop.h"
#include "typing.h"
#include <numeric>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <iostream>
#include <omp.h>

namespace lattica_hw_api {

template <typename T, typename AGetter, typename BGetter, typename CombineOp>
void elementwise_modred(
    AGetter get_a,
    BGetter get_b,
    std::shared_ptr<DeviceTensor<T>>& result,
    CombineOp combine_op)
{
    const auto& out_shape = result->dims;
    int64_t ndim = out_shape.size();

    // Compute total size and strides (same as before)…
    int64_t total = 1;
    for (auto d : out_shape) total *= d;
    std::vector<int64_t> strides(ndim,1);
    for (int i = ndim-2; i>=0; --i)
        strides[i] = strides[i+1] * out_shape[i+1];

    #pragma omp parallel for
    for (int64_t idx = 0; idx < total; ++idx) {
        std::vector<int64_t> coord(ndim);
        int64_t linear = idx;
        for (int d = 0; d < ndim; ++d) {
            coord[d]  = linear / strides[d];
            linear   %= strides[d];
        }
        T a_val = get_a(coord);
        T b_val = get_b(coord);
        result->at(coord) = combine_op(a_val, b_val);
    }
}

template <typename T, typename AGetter, typename BGetter, typename PGetter, typename CombineOp>
void elementwise_modop(
    AGetter get_a,
    BGetter get_b,
    PGetter get_p,
    std::shared_ptr<DeviceTensor<T>>& result,
    CombineOp combine_op)
{
    const auto& out_shape = result->dims;
    const int64_t ndim = out_shape.size();

    // Total number of elements
    int64_t total = 1;
    for (auto d : out_shape) total *= d;

    // Precompute strides for linear → nd coord mapping
    std::vector<int64_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * out_shape[i + 1];
    }

    #pragma omp parallel for
    for (int64_t idx = 0; idx < total; ++idx) {
        // Compute multi-dimensional coordinate
        std::vector<int64_t> coord(ndim);
        int64_t linear = idx;
        for (int d = 0; d < ndim; ++d) {
            coord[d] = linear / strides[d];
            linear %= strides[d];
        }

        T a_val = get_a(coord);
        T b_val = get_b(coord);
        T p_val = get_p(coord);
        result->at(coord) = combine_op(a_val, b_val, p_val);
    }
}


// ---- Wrapper Functions ----
#define CHECK_DIMS_MATCH_LAST(tensor, result, label) \
    if (tensor->dims.size() != 1 && tensor->dims.back() != result->dims.back()) { \
        throw std::invalid_argument("Last dimension of " label " must match last dimension of result."); \
    }

#define CHECK_DIMS_BROADCASTABLE(tensor, result, label) \
    { \
        auto td = tensor->dims; \
        auto rd = result->dims; \
        if (td.size() > rd.size()) throw std::invalid_argument(label " has more dims than result."); \
        for (int64_t i = 1; i <= static_cast<int64_t>(td.size()); ++i) { \
            int64_t t_dim = td[td.size() - i]; \
            int64_t r_dim = rd[rd.size() - i]; \
            if (t_dim != 1 && t_dim != r_dim) { \
                throw std::invalid_argument(label " not broadcast-compatible with result."); \
            } \
        } \
    }

#define CHECK_NOT_NULL(ptr, label) \
    if (!(ptr)) \
        throw std::invalid_argument(std::string(label) + \
            " pointer must not be null.");

#define DEFINE_MODULAR_ARITHMETIC_WRAPPER(OPNAME, OPERATOR) \
template <typename T> \
void OPNAME##_ttt( \
    const std::shared_ptr<DeviceTensor<T>>& a, \
    const std::shared_ptr<DeviceTensor<T>>& b, \
    const std::shared_ptr<DeviceTensor<T>>& p, \
    std::shared_ptr<DeviceTensor<T>>& result) { \
    CHECK_DIMS_BROADCASTABLE(a, result, "a"); \
    CHECK_DIMS_BROADCASTABLE(b, result, "b"); \
    CHECK_DIMS_MATCH_LAST(p, result, "p"); \
    elementwise_modop<T>( \
        [a](const std::vector<int64_t>& coord) { return a->at_with_broadcast(coord); }, \
        [b](const std::vector<int64_t>& coord) { return b->at_with_broadcast(coord); }, \
        [p](const std::vector<int64_t>& coord) { return p->at_with_broadcast(coord); }, \
        result, \
        [](T a, T b, T p) { \
            T_DP<T> tmp = OPERATOR; \
            return static_cast<T>(tmp % static_cast<T_DP<T>>(p)); \
        }); \
} \
template <typename T> \
void OPNAME##_ttc( \
    const std::shared_ptr<DeviceTensor<T>>& a, \
    const std::shared_ptr<DeviceTensor<T>>& b, \
    T p_scalar, \
    std::shared_ptr<DeviceTensor<T>>& result) { \
    CHECK_DIMS_BROADCASTABLE(a, result, "a"); \
    CHECK_DIMS_BROADCASTABLE(b, result, "b"); \
    elementwise_modop<T>( \
        [a](const std::vector<int64_t>& coord) { return a->at_with_broadcast(coord); }, \
        [b](const std::vector<int64_t>& coord) { return b->at_with_broadcast(coord); }, \
        [&](const std::vector<int64_t>&) { return p_scalar; }, \
        result, \
        [](T a, T b, T p) { \
            T_DP<T> tmp = OPERATOR; \
            return static_cast<T>(tmp % static_cast<T_DP<T>>(p)); \
        }); \
} \
template <typename T> \
void OPNAME##_tct( \
    const std::shared_ptr<DeviceTensor<T>>& a, \
    T b_scalar, \
    const std::shared_ptr<DeviceTensor<T>>& p, \
    std::shared_ptr<DeviceTensor<T>>& result) { \
    CHECK_DIMS_BROADCASTABLE(a, result, "a"); \
    CHECK_DIMS_MATCH_LAST(p, result, "p"); \
    elementwise_modop<T>( \
        [a](const std::vector<int64_t>& coord) { return a->at_with_broadcast(coord); }, \
        [&](const std::vector<int64_t>&) { return b_scalar; }, \
        [p](const std::vector<int64_t>& coord) { return p->at_with_broadcast(coord); }, \
        result, \
        [](T a, T b, T p) { \
            T_DP<T> tmp = OPERATOR; \
            return static_cast<T>(tmp % static_cast<T_DP<T>>(p)); \
        }); \
} \
template <typename T> \
void OPNAME##_tcc( \
    const std::shared_ptr<DeviceTensor<T>>& a, \
    T b_scalar, \
    T p_scalar, \
    std::shared_ptr<DeviceTensor<T>>& result) { \
    CHECK_DIMS_BROADCASTABLE(a, result, "a"); \
    elementwise_modop<T>(  \
        [a](const std::vector<int64_t>& coord) { return a->at_with_broadcast(coord); }, \
        [&](const std::vector<int64_t>&) { return b_scalar; }, \
        [&](const std::vector<int64_t>&) { return p_scalar; }, \
        result, \
        [](T a, T b, T p) { \
            T_DP<T> tmp = OPERATOR; \
            return static_cast<T>(tmp % static_cast<T_DP<T>>(p)); \
        }); \
}


#define CHECK_SAME_DIMS(tensor, result, label) \
    if ((tensor)->dims != (result)->dims) \
        throw std::invalid_argument(std::string(label) + \
            " must have exactly the same shape as result."); \

#define DEFINE_SIMPLE_MOD_WRAPPER(OPNAME) \
template <typename T> \
void OPNAME##_tt( \
    const std::shared_ptr<DeviceTensor<T>>& a, \
    const std::shared_ptr<DeviceTensor<T>>& b, \
    std::shared_ptr<DeviceTensor<T>>& result) \
{ \
    CHECK_NOT_NULL(a, "a"); \
    CHECK_NOT_NULL(b, "b"); \
    CHECK_SAME_DIMS(a, result, "a"); \
    CHECK_SAME_DIMS(b, result, "b"); \
    elementwise_modred<T>( \
        [a](auto& coord) { return a->at(coord); }, \
        [b](auto& coord) { return b->at(coord); }, \
        result, \
        [](T a, T b) { return static_cast<T>(a % b); } \
    ); \
} \
template <typename T> \
void OPNAME##_tc( \
    const std::shared_ptr<DeviceTensor<T>>& a, \
    int64_t b_scalar, \
    std::shared_ptr<DeviceTensor<T>>& result) \
{ \
    CHECK_NOT_NULL(a, "a"); \
    CHECK_SAME_DIMS(a, result, "a"); \
    elementwise_modred<T>( \
        [a](auto& coord) { return a->at(coord); }, \
        [&](auto&) { return static_cast<T>(b_scalar); }, \
        result, \
        [](T a, T b) { return static_cast<T>(a % b); } \
    ); \
} \
template <typename T> \
void OPNAME##_ct( \
    int64_t a_scalar, \
    const std::shared_ptr<DeviceTensor<T>>& b, \
    std::shared_ptr<DeviceTensor<T>>& result) \
{ \
    CHECK_NOT_NULL(b, "b"); \
    CHECK_SAME_DIMS(b, result, "b"); \
    elementwise_modred<T>( \
        [&](auto&)      { return static_cast<T>(a_scalar); }, \
        [b](auto& coord) { return b->at(coord); }, \
        result, \
        [](T a, T b) { return static_cast<T>(a % b); } \
    ); \
}


DEFINE_MODULAR_ARITHMETIC_WRAPPER(modsum, static_cast<T_DP<T>>(a) + static_cast<T_DP<T>>(b))
DEFINE_MODULAR_ARITHMETIC_WRAPPER(modmul, static_cast<T_DP<T>>(a) * static_cast<T_DP<T>>(b))
DEFINE_SIMPLE_MOD_WRAPPER(mod)

// Explicit instantiations
#define INSTANTIATE_ALL(T) \
template void modsum_ttt<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void modsum_ttc<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, T, std::shared_ptr<DeviceTensor<T>>&); \
template void modsum_tct<T>(const std::shared_ptr<DeviceTensor<T>>&, T, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void modsum_tcc<T>(const std::shared_ptr<DeviceTensor<T>>&, T, T, std::shared_ptr<DeviceTensor<T>>&); \
template void modmul_ttt<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void modmul_ttc<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, T, std::shared_ptr<DeviceTensor<T>>&); \
template void modmul_tct<T>(const std::shared_ptr<DeviceTensor<T>>&, T, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void modmul_tcc<T>(const std::shared_ptr<DeviceTensor<T>>&, T, T, std::shared_ptr<DeviceTensor<T>>&); \
template void mod_tt<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void mod_tc<T>(const std::shared_ptr<DeviceTensor<T>>&, int64_t, std::shared_ptr<DeviceTensor<T>>&); \
template void mod_ct<T>(int64_t, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \

INSTANTIATE_ALL(int32_t)
INSTANTIATE_ALL(int64_t)

} // namespace lattica_hw_api
