#include "device_memory_impl.h"
#include "ntt.h"
#include "typing.h"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <omp.h>

namespace lattica_hw_api {

namespace {

// Validate and extract dimensions from a [l, m, r, k] or [l, r, k, m] tensor
template <typename T, typename U>
void validate_ntt_inputs(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<U>>& p,
    const std::shared_ptr<DeviceTensor<U>>& perm,
    const std::shared_ptr<DeviceTensor<U>>& twiddles,
    const std::shared_ptr<DeviceTensor<U>>& result,
    int64_t& l, int64_t& m, int64_t& r, int64_t& k,
    int64_t axis
) {
    if (a->dims.size() != 4)
        throw std::invalid_argument("Input tensor 'a' must have shape [l, m, r, k] or [l, r, k, m].");

    if (axis == -1) {
        l = a->dims[0];
        m = a->dims[3];
        r = a->dims[1];
        k = a->dims[2];
    } else if (axis == -3) {
        l = a->dims[0];
        m = a->dims[1];
        r = a->dims[2];
        k = a->dims[3];
    } else {
        throw std::invalid_argument("Axis must be -1 or -3 for NTT.");
    }

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
    int64_t l, int64_t r, int64_t k, int64_t m,
    int64_t axis
) {

    if (axis == -1) {
        for (int64_t i = 0; i < l; ++i) {
            for (int64_t j = 0; j < r; ++j) {
                for (int64_t t = 0; t < k; ++t) {
                    std::vector<T> temp(m);
                    for (int64_t u = 0; u < m; ++u) {
                        int64_t pu = perm->at({u});
                        temp[u] = result->at({i, j, t, pu});
                    }
                    for (int64_t u = 0; u < m; ++u) {
                        result->at({i, j, t, u}) = temp[u];
                    }
                }
            }
        }
    } else if (axis == -3) {
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
    } else {
        throw std::invalid_argument("Axis must be -1 or -3 for permutation.");
    }
}

} // namespace

template <typename T, typename U>
void ntt(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<U>>& p,
    const std::shared_ptr<DeviceTensor<U>>& perm,
    const std::shared_ptr<DeviceTensor<U>>& twiddles,
    const std::shared_ptr<DeviceTensor<U>>& log2p_list,
    const std::shared_ptr<DeviceTensor<U>>& mu_list,
    int64_t axis,
    bool skip_perm,
    std::shared_ptr<DeviceTensor<U>>& result
) {
    int64_t l, m, r, k;
    validate_ntt_inputs<T, U>(a, p, perm, twiddles, result, l, m, r, k, axis);

    if (axis == -1) {
        #pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < l; ++i) {
            for (int64_t j = 0; j < r; ++j) {
                for (int64_t t = 0; t < k; ++t) {
                    U mod = p->at({t});

                    // Copy input to output with cast
                    for (int64_t u = 0; u < m; ++u) {
                        result->at({i, j, t, u}) = static_cast<U>(a->at({i, j, t, u}));
                    }

                    int64_t n = m;
                    int64_t step = n;
                    for (int64_t stage = 1; stage < n; stage *= 2) {
                        step /= 2;
                        for (int64_t u = 0; u < stage; ++u) {
                            int64_t j1 = 2 * u * step;
                            int64_t j2 = j1 + step;
                            U s = twiddles->at({t, stage + u});

                            for (int64_t jx = j1; jx < j2; ++jx) {
                                U u_val = result->at({i, j, t, jx});
                                U v_val = result->at({i, j, t, jx + step});
                                auto v_tw = static_cast<typename std::common_type<U, U>::type>(v_val) *
                                            static_cast<typename std::common_type<U, U>::type>(s);
                                U v_mod = static_cast<U>(v_tw % static_cast<typename std::common_type<U, U>::type>(mod));
                                result->at({i, j, t, jx}) = (u_val + v_mod) % mod;
                                result->at({i, j, t, jx + step}) = (u_val + mod - v_mod) % mod;
                            }
                        }
                    }
                }
            }
        }
    } else if (axis == -3) {
        #pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < l; ++i) {
            for (int64_t j = 0; j < r; ++j) {
                for (int64_t t = 0; t < k; ++t) {
                    U mod = p->at({t});

                    // Copy input to output with cast (index order changed)
                    for (int64_t u = 0; u < m; ++u) {
                        result->at({i, u, j, t}) = static_cast<U>(a->at({i, u, j, t}));
                    }

                    int64_t n = m;
                    int64_t step = n;
                    for (int64_t stage = 1; stage < n; stage *= 2) {
                        step /= 2;
                        for (int64_t u = 0; u < stage; ++u) {
                            int64_t j1 = 2 * u * step;
                            int64_t j2 = j1 + step;
                            U s = twiddles->at({t, stage + u});

                            for (int64_t jx = j1; jx < j2; ++jx) {
                                U u_val = result->at({i, jx, j, t});
                                U v_val = result->at({i, jx + step, j, t});
                                auto v_tw = static_cast<typename std::common_type<U, U>::type>(v_val) *
                                            static_cast<typename std::common_type<U, U>::type>(s);
                                U v_mod = static_cast<U>(v_tw % static_cast<typename std::common_type<U, U>::type>(mod));
                                result->at({i, jx, j, t}) = (u_val + v_mod) % mod;
                                result->at({i, jx + step, j, t}) = (u_val + mod - v_mod) % mod;
                            }
                        }
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument("Axis must be -1 or -3 for NTT.");
    }

    if (!skip_perm) {
        apply_permutation<U>(perm, result, l, r, k, m, axis);
    }
}


template <typename T>
void intt(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& p,
    const std::shared_ptr<DeviceTensor<T>>& perm,
    const std::shared_ptr<DeviceTensor<T>>& inv_twiddles, // now [k, m]
    const std::shared_ptr<DeviceTensor<T>>& m_inv,
    const std::shared_ptr<DeviceTensor<T>>& log2p_list,
    const std::shared_ptr<DeviceTensor<T>>& mu_list,
    std::shared_ptr<DeviceTensor<T>>& result
) {
    int64_t l, m, r, k;
    validate_ntt_inputs<T>(a, p, perm, inv_twiddles, result, l, m, r, k, -3);

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
// Explicit instantiations with optional Barrett‐reduction parameters

template void ntt<int8_t, int64_t>(
    const std::shared_ptr<DeviceTensor<int8_t>>& /*a*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*p*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*perm*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*twiddles*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*log2p_list*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*mu_list*/,
    int64_t /*axis*/,
    bool /*skip_perm*/,
    std::shared_ptr<DeviceTensor<int64_t>>& /*result*/);

template void ntt<int8_t, int32_t>(
    const std::shared_ptr<DeviceTensor<int8_t>>& /*a*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*p*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*perm*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*twiddles*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*log2p_list*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*mu_list*/,
    int64_t /*axis*/,
    bool /*skip_perm*/,
    std::shared_ptr<DeviceTensor<int32_t>>& /*result*/);

template void ntt<int32_t, int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>& /*a*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*p*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*perm*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*twiddles*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*log2p_list*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*mu_list*/,
    int64_t /*axis*/,
    bool /*skip_perm*/,
    std::shared_ptr<DeviceTensor<int32_t>>& /*result*/);

template void ntt<int64_t, int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>& /*a*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*p*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*perm*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*twiddles*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*log2p_list*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*mu_list*/,
    int64_t /*axis*/,
    bool /*skip_perm*/,
    std::shared_ptr<DeviceTensor<int64_t>>& /*result*/);

template void intt<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>& /*a*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*p*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*perm*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*inv_twiddles*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*m_inv*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*log2p_list*/,
    const std::shared_ptr<DeviceTensor<int32_t>>& /*mu_list*/,
    std::shared_ptr<DeviceTensor<int32_t>>& /*result*/);

template void intt<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>& /*a*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*p*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*perm*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*inv_twiddles*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*m_inv*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*log2p_list*/,
    const std::shared_ptr<DeviceTensor<int64_t>>& /*mu_list*/,
    std::shared_ptr<DeviceTensor<int64_t>>& /*result*/);


} // namespace lattica_hw_api
