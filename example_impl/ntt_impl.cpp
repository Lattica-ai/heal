#include "device_memory_impl.h"
#include "ntt.h"
#include "typing.h"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <omp.h>

namespace lattica_hw_api {

namespace {

// Validate and extract dimensions from a [l, m, r, k] tensor
template <typename T>
void validate_ntt_inputs(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& p,
    const std::shared_ptr<DeviceTensor<T>>& perm,
    const std::shared_ptr<DeviceTensor<T>>& twiddles,
    const std::shared_ptr<DeviceTensor<T>>& result,
    int64_t& l, int64_t& m, int64_t& r, int64_t& k
) {
    if (a->dims.size() != 4)
        throw std::invalid_argument("Input tensor 'a' must have shape [l, m, r, k].");

    l = a->dims[0];
    m = a->dims[1];
    r = a->dims[2];
    k = a->dims[3];

    if (result->dims != a->dims)
        throw std::invalid_argument("Output tensor must have the same shape as input tensor.");

    if (p->dims.size() != 1 || p->dims[0] != k)
        throw std::invalid_argument("Tensor 'p' must have shape [k].");

    if (perm->dims.size() != 1 || perm->dims[0] != m)
        throw std::invalid_argument("Tensor 'perm' must have shape [m].");

    if (twiddles->dims.size() != 2 || twiddles->dims[0] != k || twiddles->dims[1] != m)
        throw std::invalid_argument("Tensor 'twiddles' must have shape [k, m].");
}

template <typename T>
void apply_permutation(
    const std::shared_ptr<DeviceTensor<T>>& perm,
    std::shared_ptr<DeviceTensor<T>>& result,
    int64_t l, int64_t r, int64_t k, int64_t m
) {
    for (int64_t i = 0; i < l; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            for (int64_t t = 0; t < k; ++t) {
                std::vector<T> temp(m);
                for (int64_t u = 0; u < m; ++u) {
                    int64_t pu = perm->at({u});
                    temp[u] = result->at({i, pu, j, t});
                }
                for (int64_t u = 0; u < m; ++u) {
                    result->at({i, u, j, t}) = temp[u];
                }
            }
        }
    }
}

} // namespace

template <typename T>
void ntt(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& p,
    const std::shared_ptr<DeviceTensor<T>>& perm,
    const std::shared_ptr<DeviceTensor<T>>& twiddles, // now [k, m]
    std::shared_ptr<DeviceTensor<T>>& result
) {
    int64_t l, m, r, k;
    validate_ntt_inputs<T>(a, p, perm, twiddles, result, l, m, r, k);

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < l; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            for (int64_t t = 0; t < k; ++t) {
                T mod = p->at({t});

                for (int64_t u = 0; u < m; ++u) {
                    result->at({i, u, j, t}) = a->at({i, u, j, t});
                }

                int64_t n = m;
                int64_t step = n;
                for (int64_t stage = 1; stage < n; stage *= 2) {
                    step /= 2;
                    for (int64_t u = 0; u < stage; ++u) {
                        int64_t j1 = 2 * u * step;
                        int64_t j2 = j1 + step;
                        T s = twiddles->at({t, stage + u});

                        for (int64_t jx = j1; jx < j2; ++jx) {
                            T u_val = result->at({i, jx, j, t});
                            T v_val = result->at({i, jx + step, j, t});
                            T_DP<T> v_tw = static_cast<T_DP<T>>(v_val) * static_cast<T_DP<T>>(s);
                            T v_mod = static_cast<T>(v_tw % static_cast<T_DP<T>>(mod));
                            result->at({i, jx, j, t}) = (u_val + v_mod) % mod;
                            result->at({i, jx + step, j, t}) = (u_val + mod - v_mod) % mod;
                        }
                    }
                }
            }
        }
    }

    apply_permutation<T>(perm, result, l, r, k, m);
}

template <typename T>
void intt(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& p,
    const std::shared_ptr<DeviceTensor<T>>& perm,
    const std::shared_ptr<DeviceTensor<T>>& inv_twiddles, // now [k, m]
    const std::shared_ptr<DeviceTensor<T>>& m_inv,
    std::shared_ptr<DeviceTensor<T>>& result
) {
    int64_t l, m, r, k;
    validate_ntt_inputs<T>(a, p, perm, inv_twiddles, result, l, m, r, k);

    for (int64_t i = 0; i < l; ++i) {
        for (int64_t j = 0; j < r; ++j) {
            for (int64_t t = 0; t < k; ++t) {
                T mod = p->at({t});
                T m_inv_t = m_inv->at({t});

                for (int64_t u = 0; u < m; ++u) {
                    int64_t pu = perm->at({u});
                    result->at({i, pu, j, t}) = a->at({i, u, j, t});
                }

                int64_t n = m, t_stride = 1, half = n / 2;
                while (half >= 1) {
                    for (int64_t tid = 0; tid < n / 2; ++tid) {
                        int64_t group = tid / t_stride;
                        int64_t idx_u = group * t_stride * 2 + (tid % t_stride);
                        int64_t idx_v = idx_u + t_stride;
                        int64_t idx_psi = half + group;

                        T u_val = result->at({i, idx_u, j, t});
                        T v_val = result->at({i, idx_v, j, t});
                        T s = inv_twiddles->at({t, idx_psi});

                        result->at({i, idx_u, j, t}) = (u_val + v_val) % mod;
                        T_DP<T> diff = static_cast<T_DP<T>>(u_val + mod - v_val) * static_cast<T_DP<T>>(s);
                        result->at({i, idx_v, j, t}) = static_cast<T>(diff % static_cast<T_DP<T>>(mod));
                    }
                    t_stride *= 2;
                    half /= 2;
                }

                for (int64_t u = 0; u < m; ++u) {
                    T val = result->at({i, u, j, t});
                    result->at({i, u, j, t}) = (val * m_inv_t) % mod;
                }
            }
        }
    }
}

// Explicit instantiations
template void ntt<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, std::shared_ptr<DeviceTensor<int32_t>>&);
template void ntt<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&, const std::shared_ptr<DeviceTensor<int64_t>>&, const std::shared_ptr<DeviceTensor<int64_t>>&, const std::shared_ptr<DeviceTensor<int64_t>>&, std::shared_ptr<DeviceTensor<int64_t>>&);
template void intt<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, const std::shared_ptr<DeviceTensor<int32_t>>&, std::shared_ptr<DeviceTensor<int32_t>>&);
template void intt<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&, const std::shared_ptr<DeviceTensor<int64_t>>&, const std::shared_ptr<DeviceTensor<int64_t>>&, const std::shared_ptr<DeviceTensor<int64_t>>&, const std::shared_ptr<DeviceTensor<int64_t>>&, std::shared_ptr<DeviceTensor<int64_t>>&);

} // namespace lattica_hw_api
