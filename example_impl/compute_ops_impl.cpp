#include "device_memory_impl.h"
#include "compute_ops.h"
#include <stdexcept>

namespace lattica_hw_api {

    template <typename T>
    void take_along_axis(
        const std::shared_ptr<DeviceTensor<T>>& a,
        const std::shared_ptr<DeviceTensor<IndexType>>& indices,
        IndexType axis,
        std::shared_ptr<DeviceTensor<T>>& result
    ) {
        const size_t rank = a->dims.size();

        // 1) Normalize & validate axis
        if (axis < -static_cast<IndexType>(rank) || axis >= static_cast<IndexType>(rank)) {
            throw std::out_of_range("Axis out of range");
        }
        if (axis < 0) {
            axis += static_cast<IndexType>(rank);
        }

        // 2) Rank‐match check
        if (indices->dims.size() != rank) {
            throw std::invalid_argument("Indices tensor rank must match input rank");
        }

        // 3) Shape‐check on non‐axis dims
        for (size_t i = 0; i < rank; ++i) {
            if (i != static_cast<size_t>(axis) && a->dims[i] != indices->dims[i]) {
                throw std::invalid_argument("Shape mismatch at non-axis dimension");
            }
        }

        // 4) Compute number of output elements from indices shape
        int64_t total = 1;
        for (size_t i = 0; i < rank; ++i) {
            total *= indices->dims[i];
        }

        // 5) Build row‑major strides for unraveling an index into the indices‑shape
        std::vector<int64_t> out_strides(rank);
        int64_t s = 1;
        for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
            out_strides[i] = s;
            s *= indices->dims[i];
        }

        // 6) Pointers into your flat buffers
        T*          a_data    = static_cast<T*>(a->data.get());
        IndexType*  idx_data  = static_cast<IndexType*>(indices->data.get());
        T*          out_data  = static_cast<T*>(result->data.get());

        std::vector<int64_t> idx_full(rank), idx_src(rank);

        // 7) Iterate over every output element in flat (1D) index space
        for (int64_t flat = 0; flat < total; ++flat) {
            int64_t rem = flat;

            // 8) Convert the flat output index into a multi-dimensional index (row-major unraveling)
            //    This gives us the coordinate in the `indices` tensor that we're currently evaluating.
            for (size_t i = 0; i < rank; ++i) {
                idx_full[i] = rem / out_strides[i];  // compute index in dimension i
                rem %= out_strides[i];
            }

            // 9) Fetch the “select” index from the indices buffer at this multi-dimensional coordinate
            int64_t idx_offset = 0;
            for (size_t i = 0; i < rank; ++i) {
                idx_offset += idx_full[i] * indices->strides[i];
            }
            IndexType sel = idx_data[idx_offset];

            // 10) Handle negative indices (Python-style)
            if (sel < 0) {
                sel += static_cast<IndexType>(a->dims[axis]);
            }
            if (sel < 0 || sel >= static_cast<IndexType>(a->dims[axis])) {
                throw std::out_of_range("Index out of range");
            }

            // 11) Build the source coordinate in `a` by copying idx_full and
            //     replacing the value along the `axis` dimension with `sel`
            for (size_t i = 0; i < rank; ++i) {
                idx_src[i] = idx_full[i];
            }
            idx_src[axis] = sel;

            // 12) Compute the flattened offset in `a` corresponding to this coordinate
            int64_t src_offset = 0;
            for (size_t i = 0; i < rank; ++i) {
                src_offset += idx_src[i] * a->strides[i];
            }

            // 13) Write the gathered value into the result tensor at position `flat`
            out_data[flat] = a_data[src_offset];
        }
    }


    template <typename T>
    void apply_g_decomp(
        const std::shared_ptr<DeviceTensor<T>>& a,
        int32_t                                g_exp,
        int32_t                                g_base_bits,
        std::shared_ptr<DeviceTensor<T>>&      result
    ) {
        // Validate parameters
        if (g_exp <= 0) {
            throw std::invalid_argument("apply_g_decomp: g_exp must be positive");
        }
        if (g_base_bits <= 0) {
            throw std::invalid_argument("apply_g_decomp: g_base_bits must be positive");
        }
        // Ensure base bits fit in type T
        int32_t max_bits = static_cast<int32_t>(8 * sizeof(T));
        if (g_base_bits > max_bits) {
            throw std::invalid_argument(
                "apply_g_decomp: g_base_bits must not exceed the bit width of the element type");
        }

        // Validate shapes: result dims = a.dims + [g_exp]
        const auto& in_dims  = a->dims;
        const auto& out_dims = result->dims;
        if (out_dims.size() != in_dims.size() + 1) {
            throw std::invalid_argument("apply_g_decomp: result must have one extra trailing dimension");
        }
        // The size of the extra trailing dimension must match g_exp
        if (out_dims.back() != g_exp) {
            throw std::invalid_argument(
                "apply_g_decomp: the size of the extra trailing dimension must equal g_exp");
        }
        // All preceding dims must match the input dims
        for (size_t i = 0; i < in_dims.size(); ++i) {
            if (out_dims[i] != in_dims[i]) {
                throw std::invalid_argument("apply_g_decomp: result dimensions must match input dimensions");
            }
        }

        // Compute base = 2^g_base_bits
        T base = static_cast<T>(1) << g_base_bits;

        // Prepare index vector for input
        std::vector<int64_t> in_idx(in_dims.size(), 0);

        // Compute total number of elements
        int64_t total_elems = 1;
        for (auto d : in_dims) {
            total_elems *= d;
        }

        for (int64_t linear = 0; linear < total_elems; ++linear) {
            // Convert linear index to multi-dimensional input index
            int64_t rem = linear;
            for (int i = static_cast<int>(in_dims.size()) - 1; i >= 0; --i) {
                in_idx[i] = rem % in_dims[i];
                rem /= in_dims[i];
            }

            // Read input value
            T x = a->at(in_idx);

            // For each decomposition level
            for (int32_t j = 0; j < g_exp; ++j) {
                // Compute the j-th digit via bit-shift and mask
                T digit = (x >> (j * g_base_bits)) & (base - 1);

                // Build output index: in_idx + [j] as trailing dimension
                std::vector<int64_t> out_idx = in_idx;
                out_idx.push_back(j);

                // Write digit
                result->at(out_idx) = digit;
            }
        }
    }


    template <typename T>
    void abs(
        const std::shared_ptr<DeviceTensor<T>>& a,
        std::shared_ptr<DeviceTensor<T>>&       result
    ) {
        // dims must match exactly
        if (a->dims != result->dims) {
            throw std::invalid_argument(
                "abs: tensor dims do not match"
            );
        }

        // compute total element count
        const auto& dims = a->dims;
        const size_t rank = dims.size();
        size_t numel = 1;
        for (int64_t d : dims) {
            numel *= static_cast<size_t>(d);
        }

        // buffer for multi‐dimensional index
        std::vector<int64_t> idx(rank);

        // iterate every element by converting linear index → multi‐index
        for (size_t lin = 0; lin < numel; ++lin) {
            size_t tmp = lin;
            // decode in row‐major order
            for (size_t i = rank; i-- > 0; ) {
                idx[i] = static_cast<int64_t>(tmp % static_cast<size_t>(dims[i]));
                tmp /= static_cast<size_t>(dims[i]);
            }
            // apply abs via element access
            result->at(idx) = std::abs(a->at(idx));
        }
    }


    template <typename T>
    void set_const_val(
        const std::shared_ptr<DeviceTensor<T>>& a,
        T                                       val
    ) {
        // must have a valid tensor
        if (!a) {
            throw std::invalid_argument("set_const_val: input tensor is null");
        }

        // compute total element count
        const auto& dims = a->dims;
        const size_t rank = dims.size();
        size_t numel = 1;
        for (int64_t d : dims) {
            numel *= static_cast<size_t>(d);
        }

        // buffer for multi‐dimensional index
        std::vector<int64_t> idx(rank);

        // iterate every element by converting linear index → multi‐index
        for (size_t lin = 0; lin < numel; ++lin) {
            size_t tmp = lin;
            // decode in row‐major order
            for (size_t i = rank; i-- > 0; ) {
                idx[i] = static_cast<int64_t>(tmp % static_cast<size_t>(dims[i]));
                tmp  /= static_cast<size_t>(dims[i]);
            }
            // set to constant
            a->at(idx) = val;
        }
    }

// ────────────────────────────────────────────────────
// explicit instantiations for take_along_axis
template void take_along_axis<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<int32_t>>&
);
template void take_along_axis<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<int64_t>>&
);
template void take_along_axis<float>(
    const std::shared_ptr<DeviceTensor<float>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<float>>&
);
template void take_along_axis<double>(
    const std::shared_ptr<DeviceTensor<double>>&,
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int64_t,
    std::shared_ptr<DeviceTensor<double>>&
);

// ────────────────────────────────────────────────────
// explicit instantiations for apply_g_decomp
template void apply_g_decomp<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    int32_t,
    int32_t,
    std::shared_ptr<DeviceTensor<int32_t>>&
);
template void apply_g_decomp<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    int32_t,
    int32_t,
    std::shared_ptr<DeviceTensor<int64_t>>&
);

// ────────────────────────────────────────────────────
// explicit instantiations for abs
template void abs<int32_t>(
    const std::shared_ptr<DeviceTensor<int32_t>>&,
    std::shared_ptr<DeviceTensor<int32_t>>&
);
template void abs<int64_t>(
    const std::shared_ptr<DeviceTensor<int64_t>>&,
    std::shared_ptr<DeviceTensor<int64_t>>&
);
template void abs<float>(
    const std::shared_ptr<DeviceTensor<float>>&,
    std::shared_ptr<DeviceTensor<float>>&
);
template void abs<double>(
    const std::shared_ptr<DeviceTensor<double>>&,
    std::shared_ptr<DeviceTensor<double>>&
);

// explicit instantiation for set_const_val
template void set_const_val<int32_t>(const std::shared_ptr<DeviceTensor<int32_t>>&, int32_t);
template void set_const_val<int64_t>(const std::shared_ptr<DeviceTensor<int64_t>>&, int64_t);
template void set_const_val<float  >(const std::shared_ptr<DeviceTensor<float  >>&, float  );
template void set_const_val<double >(const std::shared_ptr<DeviceTensor<double >>&, double );

} // namespace lattica_hw_api
