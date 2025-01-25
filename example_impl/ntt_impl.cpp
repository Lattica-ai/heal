#include "device_memory_impl.h"
#include "ntt.h"
#include <cmath>
#include <iostream>
#include <vector>

#include "typing.h"

namespace lattica_hw_api {

namespace {

template <typename T>
void validate_inputs(
    const std::shared_ptr<DeviceMemory<T>>& a,
    const std::shared_ptr<DeviceMemory<T>>& p,
    const std::shared_ptr<DeviceMemory<T>>& perm,
    const std::shared_ptr<DeviceMemory<T>>& twiddles,
    const std::shared_ptr<DeviceMemory<T>>& result,
    size_t& l, size_t& m, size_t& k
) {
    if (a->dimensions != 3 || a->dims.size() != 3) {
        throw std::invalid_argument("Input vector 'a' must have dimensions [l, m, k].");
    }

    l = a->dims[0];
    m = a->dims[1];
    k = a->dims[2];

    if (p->dimensions != 1 || p->dims.size() != 1 || p->dims[0] != k) {
        throw std::invalid_argument("Input vector 'p' must have dimensions [k].");
    }
    if (perm->dimensions != 1 || perm->dims.size() != 1 || perm->dims[0] != m) {
        throw std::invalid_argument("Input vector 'perm' must have dimensions [m].");
    }
    if (twiddles->dimensions != 2 || twiddles->dims.size() != 2 || twiddles->dims[0] != m || twiddles->dims[1] != k) {
        throw std::invalid_argument("Input vector 'twiddles' must have dimensions [m, k].");
    }
    if (result->dimensions != 3 || result->dims.size() != 3 || result->dims[0] != l || result->dims[1] != m || result->dims[2] != k) {
        throw std::invalid_argument("Output vector 'result' must have dimensions [l, m, k].");
    }
}

template <typename T>
void apply_permutation(
    const std::shared_ptr<DeviceMemory<T>>& perm,
    std::shared_ptr<DeviceMemory<T>>& result,
    size_t batch,
    size_t prime_index,
    size_t m
) {
    std::vector<T> temp_result(m);
    for (size_t i = 0; i < m; ++i) {
        size_t permuted_index = perm->at({i});
        temp_result[i] = result->at({batch, permuted_index, prime_index});
    }
    for (size_t i = 0; i < m; ++i) {
        result->at({batch, i, prime_index}) = temp_result[i];
    }
}

} // anonymous namespace


template <typename T>
void ntt(
    const std::shared_ptr<DeviceMemory<T>>& a,        // [l, m, k]
    const std::shared_ptr<DeviceMemory<T>>& p,        // [k]
    const std::shared_ptr<DeviceMemory<T>>& perm,     // [m]
    const std::shared_ptr<DeviceMemory<T>>& twiddles, // [m, k]
    std::shared_ptr<DeviceMemory<T>>& result          // [l, m, k] (output)
) {
    size_t l, m, k;
    validate_inputs<T>(a, p, perm, twiddles, result, l, m, k);

    for (size_t batch = 0; batch < l; ++batch) {
        for (size_t prime_index = 0; prime_index < k; ++prime_index) {
            size_t n = m;
            size_t t = n;
            T mod_prime = p->at({prime_index});

            // Copy input from 'a' to 'result'
            for (size_t i = 0; i < m; ++i) {
                result->at({batch, i, prime_index}) = a->at({batch, i, prime_index});
            }

            for (size_t m = 1; m < n; m *= 2) {
                t /= 2;
                for (size_t i = 0; i < m; ++i) {
                    size_t j_1 = 2 * i * t;
                    size_t j_2 = j_1 + t;
                    T s = twiddles->at({m + i, prime_index});

                    for (size_t j = j_1; j < j_2; ++j) {
                        T u = result->at({batch, j, prime_index});
                        T_DP<T> temp_mul = static_cast<T_DP<T>>(result->at({batch, j + t, prime_index})) * static_cast<T_DP<T>>(s);
                        T v = static_cast<T>(temp_mul % static_cast<T_DP<T>>(mod_prime));

                        result->at({batch, j, prime_index}) = (u + v) % mod_prime;
                        result->at({batch, j + t, prime_index}) = (u - v + mod_prime) % mod_prime;
                    }
                }
            }
            apply_permutation<T>(perm, result, batch, prime_index, m);
        }
    }
}
template void ntt<int32_t>(
    const std::shared_ptr<DeviceMemory<int32_t>>& a,
    const std::shared_ptr<DeviceMemory<int32_t>>& p,
    const std::shared_ptr<DeviceMemory<int32_t>>& perm,
    const std::shared_ptr<DeviceMemory<int32_t>>& twiddles,
    std::shared_ptr<DeviceMemory<int32_t>>& result
);
template void ntt<int64_t>(
    const std::shared_ptr<DeviceMemory<int64_t>>& a,
    const std::shared_ptr<DeviceMemory<int64_t>>& p,
    const std::shared_ptr<DeviceMemory<int64_t>>& perm,
    const std::shared_ptr<DeviceMemory<int64_t>>& twiddles,
    std::shared_ptr<DeviceMemory<int64_t>>& result
);

template <typename T>
void intt(
    const std::shared_ptr<DeviceMemory<T>>& a,             // [l, m, k]
    const std::shared_ptr<DeviceMemory<T>>& p,             // [k]
    const std::shared_ptr<DeviceMemory<T>>& perm,          // [m]
    const std::shared_ptr<DeviceMemory<T>>& inv_twiddles,  // [m, k]
    const std::shared_ptr<DeviceMemory<T>>& m_inv,         // [k]
    std::shared_ptr<DeviceMemory<T>>& result               // [l, m, k] (output)
) {
    size_t l, m, k;
    validate_inputs<T>(a, p, perm, inv_twiddles, result, l, m, k);

    for (size_t batch = 0; batch < l; ++batch) {
        for (size_t prime_index = 0; prime_index < k; ++prime_index) {
            size_t n = m;
            size_t t = 1;
            T mod_prime = p->at({prime_index});

            // Copy input from 'a' to 'result'
            for (size_t i = 0; i < m; ++i) {
                size_t permuted_index = perm->at({i});
                result->at({batch, permuted_index, prime_index}) = a->at({batch, i, prime_index});
            }

            t = 1;
            size_t m_half = n / 2;
            while (m_half >= 1) {
                for (size_t tid = 0; tid < n / 2; ++tid) {
                    size_t i_tid = tid / t;
                    size_t idx_u = i_tid * t + tid;
                    size_t idx_v = idx_u + t;
                    size_t idx_psi = m_half + i_tid;

                    T u = result->at({batch, idx_u, prime_index});
                    T v = result->at({batch, idx_v, prime_index});
                    T s = inv_twiddles->at({idx_psi, prime_index});

                    result->at({batch, idx_u, prime_index}) = (u + v) % mod_prime;
                    T_DP<T> temp_mul = static_cast<T_DP<T>>(u - v + mod_prime) * static_cast<T_DP<T>>(s);
                    result->at({batch, idx_v, prime_index}) = static_cast<T>(temp_mul % static_cast<T_DP<T>>(mod_prime));
                }
                t *= 2;
                m_half /= 2;
            }

            // Multiply by normalization constant
            T normalization_const = m_inv->at({prime_index});
            for (size_t i = 0; i < m; ++i) {
                result->at({batch, i, prime_index}) = (result->at({batch, i, prime_index}) * normalization_const) % mod_prime;
            }
        }
    }
}
template void intt<int32_t>(
    const std::shared_ptr<DeviceMemory<int32_t>>& a,
    const std::shared_ptr<DeviceMemory<int32_t>>& p,
    const std::shared_ptr<DeviceMemory<int32_t>>& perm,
    const std::shared_ptr<DeviceMemory<int32_t>>& inv_twiddles,
    const std::shared_ptr<DeviceMemory<int32_t>>& m_inv,
    std::shared_ptr<DeviceMemory<int32_t>>& result
);
template void intt<int64_t>(
    const std::shared_ptr<DeviceMemory<int64_t>>& a,
    const std::shared_ptr<DeviceMemory<int64_t>>& p,
    const std::shared_ptr<DeviceMemory<int64_t>>& perm,
    const std::shared_ptr<DeviceMemory<int64_t>>& inv_twiddles,
    const std::shared_ptr<DeviceMemory<int64_t>>& m_inv,
    std::shared_ptr<DeviceMemory<int64_t>>& result
);

} // namespace lattica_hw_api
