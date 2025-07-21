#include "modmul_axis_sum.h"
#include "typing.h"
#include <stdexcept>

namespace lattica_hw_api {

template <typename T>
void validate_modmul_inputs(
    const std::shared_ptr<DeviceTensor<T>>& a,      // [reps, n, sum_size, p_list_len] or [reps, sum_size, p_list_len, n]
    const std::shared_ptr<DeviceTensor<T>>& b,      // [      n, sum_size, p_list_len] or [      sum_size, p_list_len, n]
    const std::shared_ptr<DeviceTensor<T>>& p,
    const std::shared_ptr<DeviceTensor<T>>& perm,
    const std::shared_ptr<DeviceTensor<T>>& result, // [reps, n,           p_list_len] or [reps,           p_list_len, n]
    int64_t axis,
    bool apply_perm,
    int64_t& reps, int64_t& n, int64_t& sum_size, int64_t& k
) {
    if (a->dims.size() != 4)
        throw std::invalid_argument("Tensor a must have rank 4.");
    if (b->dims.size() != 3)
        throw std::invalid_argument("Tensor b must have rank 3.");
    if (result->dims.size() != 3)
        throw std::invalid_argument("Result tensor must have rank 3.");

    if (b->dims[0] != a->dims[1] ||
        b->dims[1] != a->dims[2] ||
        b->dims[2] != a->dims[3]) {
        throw std::invalid_argument(
            "Tensor b must have shape [a->dims[1], a->dims[2], a->dims[3]]"
        );
    }

    reps = a->dims[0];

    if (axis == -1) {
        sum_size = a->dims[1];
        k = a->dims[2];
        n = a->dims[3];

        if (result->dims[0] != reps ||
            result->dims[1] != k ||
            result->dims[2] != n) {
            throw std::invalid_argument(
                "Result tensor must have shape [a->dims[0], a->dims[2], a->dims[3]]"
            );
        }
    }
    else if (axis == -3) {
        n = a->dims[1];
        sum_size = a->dims[2];
        k = a->dims[3];

        if (result->dims[0] != reps ||
            result->dims[1] != n ||
            result->dims[2] != k) {
            throw std::invalid_argument(
                "Result tensor must have shape [a->dims[0], a->dims[1], a->dims[3]]"
            );
        }
    } else {
        throw std::invalid_argument("Axis must be -1 or -3 for modmul_axis_sum.");
    }

    // p shape: [k]
    if (p->dims.size() != 1 || p->dims[0] != k) {
        throw std::invalid_argument("p must be a 1D tensor of shape [k]");
    }

    // Moduli must be positive
    for (int64_t i = 0; i < k; ++i) {
        if (p->at({i}) <= 0) {
            throw std::invalid_argument("Modulus value must be positive.");
        }
    }

    // Perm
    if (apply_perm) {
        if (!perm || perm->dims.size() != 1 || perm->dims[0] != n) {
            throw std::invalid_argument("perm must be a 1D tensor matching the size of the axis");
        }
        for (int64_t i = 0; i < n; ++i) {
            int64_t idx = perm->at({i});
            if (idx < 0 || idx >= n) {
                throw std::invalid_argument("perm index out of bounds");
            }
        }
    }
}


template <typename T>
void modmul_axis_sum(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& b,
    const std::shared_ptr<DeviceTensor<T>>& p,
    const std::shared_ptr<DeviceTensor<T>>& perm,
    const std::shared_ptr<DeviceTensor<T>>& log2p_list,
    const std::shared_ptr<DeviceTensor<T>>& mu_list,
    int64_t axis,
    bool apply_perm,
    std::shared_ptr<DeviceTensor<T>>& result
) {
    // Validate and extract shape
    int64_t reps, n, sum_size, k;
    validate_modmul_inputs(a, b, p, perm, result, axis, apply_perm, reps, n, sum_size, k);

    // Lambda for modular multiplication
    auto modmul = [](T_DP<T> x, T_DP<T> y, T_DP<T> mod) -> T_DP<T> {
        return (x * y) % mod;
    };

    // Main computation
    if (axis == -1) {
        // a: [reps, sum_size, k, n], b: [sum_size, k, n], result: [reps, k, n]
        #pragma omp parallel for collapse(2)
        for (int64_t r = 0; r < reps; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                for (int64_t l = 0; l < n; ++l) {
                    T_DP<T> sum = 0;
                    int64_t l_idx = l;
                    if (apply_perm) {
                        l_idx = perm->at({l});
                    }
                    T_DP<T> pp = p->at({j});
                    for (int64_t i = 0; i < sum_size; ++i) {
                        T_DP<T> aa = a->at({r, i, j, l});
                        T_DP<T> bb = b->at({i, j, l_idx});
                        sum = (sum + modmul(aa, bb, pp)) % pp;
                    }
                    T_DP<T> prev = result->at({r, j, l_idx});
                    T_DP<T> new_val = (prev + sum) % pp;
                    result->at({r, j, l_idx}) = static_cast<T>(new_val);
                }
            }
        }
    } else {  // axis == -3
        // a: [reps, n, sum_size, k], b: [n, sum_size, k], result: [reps, n, k]
        #pragma omp parallel for collapse(2)
        for (int64_t r = 0; r < reps; ++r) {
            for (int64_t l = 0; l < n; ++l) {
                int64_t l_idx = l;
                if (apply_perm) {
                    l_idx = perm->at({l});
                }
                for (int64_t j = 0; j < k; ++j) {
                    T_DP<T> sum = 0;
                    T_DP<T> pp = p->at({j});
                    for (int64_t i = 0; i < sum_size; ++i) {
                        T_DP<T> aa = a->at({r, l, i, j});
                        T_DP<T> bb = b->at({l_idx, i, j});
                        sum = (sum + modmul(aa, bb, pp)) % pp;
                    }

                    T_DP<T> prev = result->at({r, l_idx, j});
                    T_DP<T> new_val = (prev + sum) % pp;
                    result->at({r, l_idx, j}) = static_cast<T>(new_val);
                }
            }
        }
    }
}


template void modmul_axis_sum<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    int64_t,
    bool,
    std::shared_ptr<DeviceTensor<int32_t>>&
);

template void modmul_axis_sum<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    bool,
    std::shared_ptr<DeviceTensor<int64_t>>&
);

} // namespace lattica_hw_api