#include "device_memory_impl.h"
#include "modop.h"
#include "typing.h"
#include "utils.h"
#include <numeric>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <iostream>
#include <omp.h>

namespace lattica_hw_api {

template <typename T, typename AGetter, typename BGetter, typename PGetter, typename CombineOp>
void elementwise_modop(
    AGetter get_a,
    BGetter get_b,
    PGetter get_p,
    std::shared_ptr<DeviceTensor<T>>& result,
    CombineOp combine_op)
{
    const auto& out_shape = result->dims;

    // Total number of elements
    int64_t total = device_tensor_utils::numel(out_shape);

    #pragma omp parallel for
    for (int64_t idx = 0; idx < total; ++idx) {
        // Use unravel_index to compute multi-dimensional coordinate
        std::vector<int64_t> coord = device_tensor_utils::unravel_index(idx, out_shape);

        T a_val = get_a(coord);
        T b_val = get_b(coord);
        T p_val = get_p(coord);
        result->at(coord) = combine_op(a_val, b_val, p_val);
    }
}


// ---- Wrapper Functions ----
#define CHECK_DIMS_MATCH_LAST(tensor, result, label) \
    if (tensor->dims.size() != 1 || tensor->dims.back() != result->dims.back()) { \
        throw std::invalid_argument(label " should be one-dimensional, and its size must match the last dimension of the result."); \
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

#define MAKE_MOD_COMBINE_LAMBDA(OP_EXPR)                  \
[](T a, T b, T p) {                                       \
    using Wide = T_DP<T>;                                 \
    /* do the wide‐precision op */                        \
    Wide tmp = OP_EXPR;                                   \
    /* C++ remainder, may be negative */                  \
    Wide rem = tmp % static_cast<Wide>(p);                \
    /* shift into [0,p) */                                \
    if (rem < 0) rem += static_cast<Wide>(p);             \
    return static_cast<T>(rem);                           \
}

#define DEFINE_MODOP_WRAPPER(OPNAME, OP_EXPR) \
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
        MAKE_MOD_COMBINE_LAMBDA(OP_EXPR)); \
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
        MAKE_MOD_COMBINE_LAMBDA(OP_EXPR)); \
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
        MAKE_MOD_COMBINE_LAMBDA(OP_EXPR)); \
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
        MAKE_MOD_COMBINE_LAMBDA(OP_EXPR)); \
}

DEFINE_MODOP_WRAPPER(modsum, static_cast<T_DP<T>>(a) + static_cast<T_DP<T>>(b))
DEFINE_MODOP_WRAPPER(modmul, static_cast<T_DP<T>>(a) * static_cast<T_DP<T>>(b))

// Explicit instantiations
#define INSTANTIATE_ALL(T) \
template void modsum_ttt<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void modsum_ttc<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, T, std::shared_ptr<DeviceTensor<T>>&); \
template void modsum_tct<T>(const std::shared_ptr<DeviceTensor<T>>&, T, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void modsum_tcc<T>(const std::shared_ptr<DeviceTensor<T>>&, T, T, std::shared_ptr<DeviceTensor<T>>&); \
template void modmul_ttt<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void modmul_ttc<T>(const std::shared_ptr<DeviceTensor<T>>&, const std::shared_ptr<DeviceTensor<T>>&, T, std::shared_ptr<DeviceTensor<T>>&); \
template void modmul_tct<T>(const std::shared_ptr<DeviceTensor<T>>&, T, const std::shared_ptr<DeviceTensor<T>>&, std::shared_ptr<DeviceTensor<T>>&); \
template void modmul_tcc<T>(const std::shared_ptr<DeviceTensor<T>>&, T, T, std::shared_ptr<DeviceTensor<T>>&);

INSTANTIATE_ALL(int32_t)
INSTANTIATE_ALL(int64_t)



// ---------- Modular Negation Implementations ----------

template <typename T>
void modneg_tt(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& p,
    std::shared_ptr<DeviceTensor<T>>& result)
{
    // — no broadcasting: shapes must be identical
    if (a->dims != p->dims || a->dims != result->dims) {
        throw std::invalid_argument("modneg_tt: tensor shapes of a, p and result must match exactly");
    }

    // — p must be strictly positive everywhere
    {
        auto const& shape = p->dims;
        int64_t total = device_tensor_utils::numel(shape);
        for (int64_t idx = 0; idx < total; ++idx) {
            auto coord = device_tensor_utils::unravel_index(idx, shape);
            T pv = p->at(coord);
            if (pv <= 0) {
                throw std::invalid_argument("modneg_tt: modulus p must be strictly positive");
            }
        }
    }

    // element‐wise (-a) % p
    elementwise_modop<T>(
        [a](auto const& coord) { return a->at(coord); },
        // dummy b‐getter (unused)
        [](auto const&) { return T(0); },
        [p](auto const& coord) { return p->at(coord); },
        result,
        [](T a_val, T /*unused*/, T p_val) {
            using Wide = T_DP<T>;
            Wide tmp = -static_cast<Wide>(a_val);
            Wide rem = tmp % static_cast<Wide>(p_val);
            if (rem < 0) rem += static_cast<Wide>(p_val);
            return static_cast<T>(rem);
        }
    );
}

template <typename T>
void modneg_tc(
    const std::shared_ptr<DeviceTensor<T>>& a,
    T p_scalar,
    std::shared_ptr<DeviceTensor<T>>& result)
{
    // — no broadcasting: shape of result must match a
    if (a->dims != result->dims) {
        throw std::invalid_argument("modneg_tc: tensor shape of result must match a");
    }
    // — p_scalar must be strictly positive
    if (p_scalar <= 0) {
        throw std::invalid_argument("modneg_tc: modulus p_scalar must be strictly positive");
    }

    elementwise_modop<T>(
        [a](auto const& coord) { return a->at(coord); },
        [](auto const&) { return T(0); },
        [&](auto const&) { return p_scalar; },
        result,
        [](T a_val, T /*unused*/, T p_val) {
            using Wide = T_DP<T>;
            Wide tmp = -static_cast<Wide>(a_val);
            Wide rem = tmp % static_cast<Wide>(p_val);
            if (rem < 0) rem += static_cast<Wide>(p_val);
            return static_cast<T>(rem);
        }
    );
}

template void modneg_tt<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    std::shared_ptr<DeviceTensor<int32_t>>&);
template void modneg_tt<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    std::shared_ptr<DeviceTensor<int64_t>>&);

template void modneg_tc<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    int32_t,
    std::shared_ptr<DeviceTensor<int32_t>>&);
template void modneg_tc<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<int64_t>>&);

} // namespace lattica_hw_api
