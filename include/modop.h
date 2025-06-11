#ifndef MODOP_H
#define MODOP_H

/**
 * @file modop.h
 * @brief Provides elementwise modular operations over tensors in DeviceTensor.
 *
 * This module implements functions for elementwise modular multiplication, modular
 * addition, modular negation, and modular reduction (remainder) between tensors stored in device memory.
 * Operations support broadcasting and non-contiguous layout via internal
 * stride-aware indexing.
 *
 * Operations:
 * - Multiplication: result = (a * b) % p
 * - Addition:       result = (a + b) % p
 * - Negation:       result = (-a) % p  (canonical non-negative remainder)
 * - Remainder:      result = a % b
 *
 * Common requirements:
 * - `a` must be a tensor of arbitrary shape, where the last dimension is either 1 or k.
 * - `b` and `p` (where applicable) must be tensors or scalars broadcastable to `a` (for mul/add)
 *   or exactly matching `a` (for negation/remainder tensor–tensor variants).
 * - `p` may be a 1D tensor of shape `[k]` or a scalar for mul/add or negation variants.
 * - `result` must be pre-allocated and have the broadcasted shape of the inputs (mul/add)
 *   or exactly match `a` (negation/remainder tensor–tensor).
 * - Non-contiguous tensors are supported using internal stride-aware indexing.
 *
 * Broadcasting semantics follow PyTorch’s rules (only in multiplication and addition):
 * - Dimensions are compared from the trailing dimensions backward.
 * - A dimension of size 1 can be expanded to match the other tensor’s size.
 * - Leading dimensions can be added implicitly.
 *
 * Variants for each operation:
 * - **ttt**: all inputs are tensors (broadcastable)
 * - **ttc**: first two inputs are tensors, last input is scalar
 * - **tct**: first and last inputs are tensors, middle is scalar
 * - **tcc**: first input is tensor, next two are scalars
 *
 * Additional remainder and negation variants:
 * - **mod** (remainder):
 *   - **tt**: both a and b are tensors (same shape)
 *   - **tc**: a is tensor, b is scalar
 *   - **ct**: a is scalar, b is tensor
 * - **modneg** (negation):
 *   - **tt**: a and p are tensors (same shape)
 *   - **tc**: a is tensor, p is scalar
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
        T b_scalar,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void mod_ct(
        T a_scalar,
        const std::shared_ptr<DeviceTensor<T>>& b,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    // ---------- Modular Negation Variants ----------

    template <typename T>
    void modneg_tt(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<T>>& p,
        std::shared_ptr<DeviceTensor<T>>& result
    );

    template <typename T>
    void modneg_tc(
        const std::shared_ptr<DeviceTensor<T>>& a,
        T p_scalar,
        std::shared_ptr<DeviceTensor<T>>& result
    );
}

#endif // MODOP_H
