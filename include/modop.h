#ifndef MODOP_H
#define MODOP_H

/**
 * @file modop.h
 * @brief Provides elementwise modular operations over tensors in DeviceTensor.
 *
 * This module implements functions for elementwise modular multiplication, modular
 * addition, and modular reduction (remainder) between tensors stored in device memory.
 * All operations support broadcasting and non-contiguous layout via internal
 * stride-aware indexing.
 *
 * Operations:
 * - Multiplication: result = (a * b) % p
 * - Addition:       result = (a + b) % p
 * - Remainder:      result = a % b
 *
 * Common requirements:
 * - `a` must be a tensor of arbitrary shape, where the last dimension is either 1 or k.
 * - `b` and `p` (where applicable) must be tensors or scalars broadcastable to `a`.
 * - `p` may be a 1D tensor of shape `[k]` or a scalar for mul/add variants.
 * - `result` must be preallocated and have the broadcasted shape of the inputs.
 * - All inputs must be of the same element type `T` (except scalar remainders use int64_t).
 * - Non-contiguous tensors are supported using internal stride-aware indexing.
 *
 * Broadcasting semantics follow PyTorch's rules:
 * - Dimensions are compared from the trailing dimensions backward.
 * - A dimension of size 1 can be expanded to match the other tensor's size.
 * - Leading dimensions can be added implicitly.
 *
 * Variants for each operation:
 * - ttt: all inputs are tensors (broadcastable)
 * - ttc: first two inputs are tensors, last input is scalar
 * - tct: first and last inputs are tensors, middle is scalar
 * - tcc: first input is tensor, next two are scalars
 *
 * Additional remainder variants (modulus):
 * - tt: both a and b are tensors
 * - tc: a is tensor, b is scalar
 * - ct: a is scalar, b is tensor
 */

namespace lattica_hw_api {

    // ---------- Modular Multiplication Variants ----------

    template <typename T>
    void modmul_ttt(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<T>>& b,
        const std::shared_ptr<DeviceTensor<T>>& p,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void modmul_ttc(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<T>>& b,
        T p_scalar,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void modmul_tct(
        const std::shared_ptr<DeviceTensor<T>>& a,
        T b_scalar,
        const std::shared_ptr<DeviceTensor<T>>& p,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void modmul_tcc(
        const std::shared_ptr<DeviceTensor<T>>& a,
        T b_scalar,
        T p_scalar,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    // ---------- Modular Addition Variants ----------

    template <typename T>
    void modsum_ttt(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<T>>& b,
        const std::shared_ptr<DeviceTensor<T>>& p,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void modsum_ttc(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<T>>& b,
        T p_scalar,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void modsum_tct(
        const std::shared_ptr<DeviceTensor<T>>& a,
        T b_scalar,
        const std::shared_ptr<DeviceTensor<T>>& p,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void modsum_tcc(
        const std::shared_ptr<DeviceTensor<T>>& a,
        T b_scalar,
        T p_scalar,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    // ---------- Modular Remainder (Modulus) Variants ----------

    template <typename T>
    void mod_tt(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<T>>& b,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void mod_tc(
        const std::shared_ptr<DeviceTensor<T>>& a,
        int64_t b_scalar,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void mod_ct(
        int64_t a_scalar,
        const std::shared_ptr<DeviceTensor<T>>& b,
        std::shared_ptr<DeviceTensor<T>>& result
    );

}

#endif // MODOP_H
