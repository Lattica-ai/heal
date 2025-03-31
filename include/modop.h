#ifndef MODOP_H
#define MODOP_H

/**
 * @file modop.h
 * @brief Provides elementwise modular operations over tensors in DeviceTensor.
 *
 * This module implements functions for elementwise modular multiplication and modular
 * addition between tensors stored in device memory. All operations support broadcasting
 * and non-contiguous layout via internal stride-aware indexing.
 *
 * Operation: result = (a OP b) % p
 *
 * Variants:
 * - ttt: a, b, p are all tensors (b and p broadcastable to a)
 * - ttc: a, b are tensors, p is a scalar
 * - tct: a and p are tensors, b is a scalar
 * - tcc: a is a tensor, b and p are scalars
 *
 * Requirements:
 * - `a` must be a tensor of arbitrary shape, where the last dimension is either 1 or k.
 * - `b` and `p` must be tensors or scalars broadcastable to `a`.
 * - `p` may be a 1D tensor of shape `[k]` or a scalar.
 * - `result` must be preallocated and have the broadcasted shape of the inputs.
 * - All inputs must be of the same element type `T`.
 * - Non-contiguous tensors are supported using internal stride-aware indexing.
 *
 * Broadcasting semantics follow PyTorch's rules:
 * - Dimensions are compared from the trailing dimensions backward.
 * - A dimension of size 1 can be expanded to match the other tensor's size.
 * - Leading dimensions can be added implicitly.
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

}

#endif // MODOP_H
