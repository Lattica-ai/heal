#include "modular_ops.h"
#include "gadget_decomposition_full_q.h"
#include "tensor_layout_ops.h"
#include "device_memory.h"
#include "tensor_value_ops.h"
#include "ntt.h"
#include "modular_axis_sum_ops.h"
#include "device_tensor_ex.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;
using namespace lattica_hw_api;

// A helper to get the type-specific suffix for function names
template <typename T>
struct TypeSuffix {
    static constexpr const char* value = "";
};
template <> struct TypeSuffix<int8_t> { static constexpr const char* value = "8"; };
template <> struct TypeSuffix<int32_t> { static constexpr const char* value = "32"; };
template <> struct TypeSuffix<int64_t> { static constexpr const char* value = "64"; };

//================================================================================
// BINDING HELPERS FOR DIFFERENT OPERATION CATEGORIES
//================================================================================

/**
 * @brief Binds the DeviceTensor class with its methods.
 */
template <typename T>
void bind_device_tensor(py::module_& m, const std::string& suffix) {
    using DeviceMem = DeviceTensor<T>;
    py::class_<DeviceMem, std::shared_ptr<DeviceMem>>(m, ("DeviceTensor" + suffix).c_str())
        .def("print", &DeviceMem::print)
        .def("print_metadata", &DeviceMem::print_metadata)
        .def("numel", &DeviceMem::numel, "Get the total number of elements in the tensor.");
}

/**
 * @brief Binds memory management functions
 * (empty, host_to_device, device_to_host, zeros, contiguous).
 */
template <typename T>
void bind_memory_ops(py::module_& m, const std::string& suffix) {
    m.def(("empty_" + suffix).c_str(), &empty<T>, py::arg("dims"),
          "Allocate a new device tensor on hardware without initializing elements.");

    m.def(("contiguous_" + suffix).c_str(), &contiguous<T>, py::arg("tensor"),
          "Return a contiguous version of the tensor.");

    m.def(("host_to_device_" + suffix).c_str(), &host_to_device<T>, py::arg("tensor"),
          "Upload a PyTorch tensor to device memory.");

    m.def(("device_to_host_" + suffix).c_str(), &device_to_host<T>, py::arg("memory"),
          "Download a device tensor back into a torch::Tensor.");

    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
        m.def(("zeros_" + suffix).c_str(), &zeros<T>, py::arg("dims"),
            "Allocate a new device tensor on hardware with all elements initialized to zero.");
    }
}

/**
 * @brief Binds all tensor layout and manipulation operations
 * (reshape, moveaxis, expand, squeeze, unsqueeze, flatten, get_slice).
 */
template <typename T>
void bind_tensor_layout_ops(py::module_& m, const std::string& suffix) {
    m.def(("reshape_" + suffix).c_str(), &reshape<T>, py::arg("tensor"), py::arg("new_shape"),
          "Reshape a tensor to the specified shape without changing its data.");

    m.def(("moveaxis_" + suffix).c_str(), &moveaxis<T>, py::arg("tensor"), py::arg("axis_src"), py::arg("axis_dst"),
          "Move an existing dimension from axis_src to axis_dst in-place.");

    m.def(("expand_" + suffix).c_str(), &expand<T>, py::arg("tensor"), py::arg("axis"), py::arg("repeats"),
          "Expand a tensor by repeating elements along the specified axis.");

    m.def(("flatten_" + suffix).c_str(), &flatten<T>, py::arg("tensor"), py::arg("start_axis"), py::arg("end_axis"),
          "Flatten the tensor between start_axis and end_axis, preserving other dimensions.");

    m.def(("get_slice_" + suffix).c_str(),
        [](const std::shared_ptr<DeviceTensor<T>>& tensor, py::iterable sliceList) {
            std::vector<SliceArg> args;
            for (py::handle h : sliceList) {
                if (py::isinstance<py::int_>(h)) {
                    args.emplace_back(h.cast<int64_t>());
                } else if (py::isinstance<py::slice>(h)) {
                    py::slice sl = h.cast<py::slice>();
                    args.emplace_back(Slice(
                        sl.attr("start").cast<int64_t>(),
                        sl.attr("stop").cast<int64_t>(),
                        sl.attr("step").cast<int64_t>()
                    ));
                } else {
                    throw std::invalid_argument("get_slice: sliceList must contain only ints or slices");
                }
            }
            return get_slice<T>(tensor, args);
        },
        py::arg("tensor"), py::arg("sliceList"),
        "Returns a zero-copy view of tensor sliced along each axis.");

    if constexpr (!std::is_same_v<T, int8_t>) {
        m.def(("squeeze_" + suffix).c_str(), &squeeze<T>, py::arg("tensor"), py::arg("axis"),
              "Remove a dimension of length 1 at the specified axis.");

        m.def(("unsqueeze_" + suffix).c_str(), &unsqueeze<T>, py::arg("tensor"), py::arg("axis"),
              "Insert a new axis of length 1 at the specified position in the tensor's shape.");
    }
}

/**
 * @brief Binds all element-wise modular arithmetic operations.
 */
template <typename T>
void bind_modular_ops(py::module_& m, const std::string& suffix) {
    // Modular Multiplication
    m.def(("modmul_ttt_" + suffix).c_str(), &modmul_ttt<T>,
        py::arg("a"), py::arg("b"), py::arg("p"), py::arg("result"),
        "([...,k] * [...,k]) % [k]");
    m.def(("modmul_ttc_" + suffix).c_str(), &modmul_ttc<T>,
        py::arg("a"), py::arg("b"), py::arg("p_scalar"), py::arg("result"),
        "([...,k] * [...,k]) % scalar");
    m.def(("modmul_tct_" + suffix).c_str(), &modmul_tct<T>,
        py::arg("a"), py::arg("b_scalar"), py::arg("p"), py::arg("result"),
        "([...,k] * scalar) % [k]");
    m.def(("modmul_tcc_" + suffix).c_str(), &modmul_tcc<T>,
        py::arg("a"), py::arg("b_scalar"), py::arg("p_scalar"), py::arg("result"),
        "([...,k] * scalar) % scalar");

    // Modular Addition
    m.def(("modsum_ttt_" + suffix).c_str(), &modsum_ttt<T>,
        py::arg("a"), py::arg("b"), py::arg("p"), py::arg("result"),
        "([...,k] + [...,k]) % [k]");
    m.def(("modsum_ttc_" + suffix).c_str(), &modsum_ttc<T>,
        py::arg("a"), py::arg("b"), py::arg("p_scalar"), py::arg("result"),
        "([...,k] + [...,k]) % scalar");
    m.def(("modsum_tct_" + suffix).c_str(), &modsum_tct<T>,
        py::arg("a"), py::arg("b_scalar"), py::arg("p"), py::arg("result"),
        "([...,k] + scalar) % [k]");
    m.def(("modsum_tcc_" + suffix).c_str(), &modsum_tcc<T>,
        py::arg("a"), py::arg("b_scalar"), py::arg("p_scalar"), py::arg("result"),
        "([...,k] + scalar) % scalar");

    // Modular Remainder
    m.def(("mod_tt_" + suffix).c_str(), &mod_tt<T>, py::arg("a"), py::arg("b"), py::arg("result"), "[...,k] % [...,k]");
    m.def(("mod_tc_" + suffix).c_str(), &mod_tc<T>, py::arg("a"), py::arg("b_scalar"), py::arg("result"), "[...,k] % scalar");
    m.def(("mod_ct_" + suffix).c_str(), &mod_ct<T>, py::arg("a_scalar"), py::arg("b"), py::arg("result"), "scalar % [...,k]");

    // Modular Negation
    m.def(("modneg_tt_" + suffix).c_str(), &modneg_tt<T>, py::arg("a"), py::arg("p"), py::arg("result"), "(-[...,k]) % [...,k]");
    m.def(("modneg_tc_" + suffix).c_str(), &modneg_tc<T>, py::arg("a"), py::arg("p_scalar"), py::arg("result"), "(-[...,k]) % scalar");
}

/**
 * @brief Binds general tensor operations like axis reductions and assignments.
 */
template <typename T>
void bind_general_ops(py::module_& m, const std::string& suffix) {
    m.def(("axis_modsum_" + suffix).c_str(), &axis_modsum<T>,
        py::arg("a"), py::arg("p"), py::arg("axis"), py::arg("result"),
        "Axis-wise modular sum");

    m.def(("modmul_axis_sum_" + suffix).c_str(), &modmul_axis_sum<T>,
        py::arg("a"), py::arg("b"), py::arg("p"), py::arg("perm"), py::arg("log2p_list"), py::arg("mu_list"),
        py::arg("axis"), py::arg("apply_perm"), py::arg("result"), "Element-wise modular multiply and sum over a specified axis");

    m.def(("take_along_axis_" + suffix).c_str(), &take_along_axis<T>,
        py::arg("tensor"), py::arg("indices"), py::arg("axis"), py::arg("result"),
        "Take elements from tensor along a specified axis using indices");

    m.def(("set_const_val_" + suffix).c_str(), &set_const_val<T>,
        py::arg("tensor"), py::arg("value"), "Set all elements to a constant value");

    m.def(("pad_single_axis_" + suffix).c_str(), &pad_single_axis<T>,
        py::arg("tensor"), py::arg("pad"), py::arg("axis"), py::arg("result"), "Pad a single axis with zeros");
}

/**
 * @brief Binds g-decomposition operations.
 */
template <typename T, typename U>
void bind_g_decomp_relative_to_full_q(py::module_& m) {
    const std::string suffix = std::string(TypeSuffix<T>::value) + "_" + std::string(TypeSuffix<U>::value);
    m.def(("apply_g_decomp_relative_to_full_q_" + suffix).c_str(), &apply_g_decomp_relative_to_full_q<T,U>,
        py::arg("a"), py::arg("q_list"), py::arg("g_exp"), py::arg("g_base_bits"), py::arg("out"),
        "G decomposition relative to full q (base 2^base_bits)");
}

/**
 * @brief Binds NTT/INTT operations.
 */
template <typename T, typename U>
void bind_ntt(py::module_& m) {
    const std::string suffix = std::string(TypeSuffix<T>::value) + "_" + std::string(TypeSuffix<U>::value);
    m.def(("ntt_" + suffix).c_str(), &ntt<T, U>,
        py::arg("a"), py::arg("p"), py::arg("perm"), py::arg("twiddles"), py::arg("log2p_list"),
        py::arg("mu_list"), py::arg("axis"), py::arg("skip_perm"), py::arg("result"), "Number Theoretic Transform");
}


//================================================================================
// MAIN BINDING FUNCTION
//================================================================================

/**
 * @brief Master template function to bind all supported operations for a given type.
 */
template <typename T>
void bind_all_operations_for(py::module_& m) {
    const std::string suffix = TypeSuffix<T>::value;

    bind_device_tensor<T>(m, suffix);
    bind_memory_ops<T>(m, suffix);
    bind_tensor_layout_ops<T>(m, suffix);

    // Bind operations that are only available for 32 and 64-bit integers
    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
        bind_modular_ops<T>(m, suffix);
        bind_general_ops<T>(m, suffix);
        m.def(("intt_" + suffix).c_str(), &intt<T>,
            py::arg("a"), py::arg("p"), py::arg("perm"), py::arg("inv_twiddles"), py::arg("m_inv"),
            py::arg("log2p_list"), py::arg("mu_list"), py::arg("result"), "Inverse Number Theoretic Transform");
    }
}


PYBIND11_MODULE(lattica_hw, m) {
    m.doc() = "Lattica Hardware API Python bindings";

    // --- Bind operations for each primary data type ---
    bind_all_operations_for<int8_t>(m);
    bind_all_operations_for<int32_t>(m);
    bind_all_operations_for<int64_t>(m);

    // --- Bind multi-type operations like G-Decomposition ---
    bind_g_decomp_relative_to_full_q<int32_t, int8_t>(m);
    bind_g_decomp_relative_to_full_q<int64_t, int8_t>(m);
    bind_g_decomp_relative_to_full_q<int32_t, int32_t>(m);
    bind_g_decomp_relative_to_full_q<int64_t, int64_t>(m);

    // --- Bind multi-type operations like NTT ---
    bind_ntt<int8_t, int32_t>(m);
    bind_ntt<int8_t, int64_t>(m);
    bind_ntt<int32_t, int32_t>(m);
    bind_ntt<int64_t, int64_t>(m);
}