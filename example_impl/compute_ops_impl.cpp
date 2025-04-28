#include "device_memory_impl.h"
#include "compute_ops.h"
#include "utils.h"
#include <stdexcept>

namespace lattica_hw_api {

    template <typename T>
    void take_along_axis(
        const std::shared_ptr<DeviceTensor<T>>&        input,
        const std::shared_ptr<DeviceTensor<int64_t>>& indices,
        int64_t                                      axis,
        std::shared_ptr<DeviceTensor<T>>&              output
    ) {
        const size_t rank = input->dims.size();

        // Normalize & validate axis
        if (axis < -static_cast<int64_t>(rank) || axis >= static_cast<int64_t>(rank)) {
            throw std::out_of_range("Axis out of range");
        }
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }

        // Rank‐match check
        if (indices->dims.size() != rank) {
            throw std::invalid_argument("Indices tensor rank must match input rank");
        }

        // Shape‐check on non‐axis dims
        for (size_t i = 0; i < rank; ++i) {
            if (i != static_cast<size_t>(axis) && input->dims[i] != indices->dims[i]) {
                throw std::invalid_argument("Shape mismatch at non-axis dimension");
            }
        }

        // Compute number of output elements from indices shape
        size_t num_elements = device_tensor_utils::numel(indices->dims);

        // Grab raw pointers
        T*         input_data = static_cast<T*>(input->data.get());
        int64_t*   idx_data   = static_cast<int64_t*>(indices->data.get());
        T*         out_data   = static_cast<T*>(output->data.get());

        // Prepare index buffers
        std::vector<int64_t> out_index(rank), src_index(rank);

        // Iterate over every output element by flat index
        for (size_t flat_idx = 0; flat_idx < num_elements; ++flat_idx) {
            // Get multi‐dimensional index into indices tensor
            out_index = device_tensor_utils::unravel_index(flat_idx, indices->dims);

            // Compute offset into the indices buffer
            int64_t idx_offset = 0;
            for (size_t i = 0; i < rank; ++i) {
                idx_offset += out_index[i] * indices->strides[i];
            }
            int64_t selected_idx = idx_data[idx_offset];

            // Python‐style negative indexing
            if (selected_idx < 0) {
                selected_idx += static_cast<int64_t>(input->dims[axis]);
            }
            if (selected_idx < 0 || selected_idx >= static_cast<int64_t>(input->dims[axis])) {
                throw std::out_of_range("Index out of range");
            }

            // Build the source coordinate
            src_index = out_index;
            src_index[axis] = selected_idx;

            // Flatten that source coordinate
            int64_t src_offset = 0;
            for (size_t i = 0; i < rank; ++i) {
                src_offset += src_index[i] * input->strides[i];
            }

            // Gather the value
            out_data[flat_idx] = input_data[src_offset];
        }
    }


    template <typename T>
    void apply_g_decomp(
        const std::shared_ptr<DeviceTensor<T>>& input,
        int32_t                                g_exp,
        int32_t                                g_base_bits,
        std::shared_ptr<DeviceTensor<T>>&      output
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

        // Validate shapes: output dims = input.dims + [g_exp]
        const auto& in_dims  = input->dims;
        const auto& out_dims = output->dims;
        if (out_dims.size() != in_dims.size() + 1) {
            throw std::invalid_argument("apply_g_decomp: output must have one extra trailing dimension");
        }

        // The size of the extra trailing dimension must match g_exp
        if (out_dims.back() != g_exp) {
            throw std::invalid_argument(
                "apply_g_decomp: the size of the extra trailing dimension must equal g_exp");
        }

        // All preceding dims must match the input dims
        for (size_t i = 0; i < in_dims.size(); ++i) {
            if (out_dims[i] != in_dims[i]) {
                throw std::invalid_argument("apply_g_decomp: output dimensions must match input dimensions");
            }
        }

        // Compute base = 2^g_base_bits
        T base = static_cast<T>(1) << g_base_bits;

        // Compute total number of elements
        size_t num_elements = device_tensor_utils::numel(in_dims);

        // Prepare index vector for input
        std::vector<int64_t> in_index(in_dims.size());

        // Iterate over every input element by flat index
        for (size_t flat_idx = 0; flat_idx < num_elements; ++flat_idx) {
            // Convert linear index to multi-dimensional input index
            in_index = device_tensor_utils::unravel_index(flat_idx, in_dims);

            // Read input value
            T value = input->at(in_index);

            // For each decomposition level
            for (int32_t level = 0; level < g_exp; ++level) {
                // Compute the digit via bit-shift and mask
                T digit = (value >> (level * g_base_bits)) & (base - 1);

                // Build output index: in_index + [level] as trailing dimension
                std::vector<int64_t> out_index = in_index;
                out_index.push_back(level);

                // Write digit
                output->at(out_index) = digit;
            }
        }
    }


    template <typename T>
    void abs(
        const std::shared_ptr<DeviceTensor<T>>& input,
        std::shared_ptr<DeviceTensor<T>>&       output
    ) {
        const auto& dims = input->dims;

        // Validate dimensions match exactly
        if (dims != output->dims) {
            throw std::invalid_argument("abs: tensor dimensions do not match");
        }

        // Compute total element count
        size_t num_elements = device_tensor_utils::numel(dims);

        // Prepare buffer for multi‐dimensional index
        std::vector<int64_t> index(dims.size());

        // Iterate over every element by flat index
        for (size_t flat_idx = 0; flat_idx < num_elements; ++flat_idx) {
            // Convert linear index to multi-dimensional index
            index = device_tensor_utils::unravel_index(flat_idx, dims);

            // Apply abs via element access
            output->at(index) = std::abs(input->at(index));
        }
    }


    template <typename T>
    void set_const_val(
        const std::shared_ptr<DeviceTensor<T>>& tensor,
        T                                       value
    ) {
        // Validate input tensor
        if (!tensor) {
            throw std::invalid_argument("set_const_val: input tensor is null");
        }

        const auto& dims = tensor->dims;

        // Compute total element count
        size_t num_elements = device_tensor_utils::numel(dims);

        // Prepare buffer for multi‐dimensional index
        std::vector<int64_t> index(dims.size());

        // Iterate over every element by flat index
        for (size_t flat_idx = 0; flat_idx < num_elements; ++flat_idx) {
            // Convert linear index to multi-dimensional index
            index = device_tensor_utils::unravel_index(flat_idx, dims);

            // Set element to constant value
            tensor->at(index) = value;
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
