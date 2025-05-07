#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "lattica_hw_api.h"

namespace py = pybind11;
using namespace lattica_hw_api;

template <typename T>
void bind_modop_variants(py::module_& m, const std::string& suffix) {
    // modmul variants
    m.def(("modmul_ttt_" + suffix).c_str(), &modmul_ttt<T>,
          "Elementwise modular multiplication: ([...,k] * [...,k]) % [k]");
    m.def(("modmul_ttc_" + suffix).c_str(), &modmul_ttc<T>,
          "Elementwise modular multiplication: ([...,k] * [...,k]) % scalar");
    m.def(("modmul_tct_" + suffix).c_str(), &modmul_tct<T>,
          "Elementwise modular multiplication: ([...,k] * scalar) % [k]");
    m.def(("modmul_tcc_" + suffix).c_str(), &modmul_tcc<T>,
          "Elementwise modular multiplication: ([...,k] * scalar) % scalar");

    // modsum variants
    m.def(("modsum_ttt_" + suffix).c_str(), &modsum_ttt<T>,
          "Elementwise modular addition: ([...,k] + [...,k]) % [k]");
    m.def(("modsum_ttc_" + suffix).c_str(), &modsum_ttc<T>,
          "Elementwise modular addition: ([...,k] + [...,k]) % scalar");
    m.def(("modsum_tct_" + suffix).c_str(), &modsum_tct<T>,
          "Elementwise modular addition: ([...,k] + scalar) % [k]");
    m.def(("modsum_tcc_" + suffix).c_str(), &modsum_tcc<T>,
          "Elementwise modular addition: ([...,k] + scalar) % scalar");
}

template <typename T>
void bind_g_decomposition(py::module_& m, const std::string& suffix) {
    m.def(("g_decomposition_" + suffix).c_str(), &g_decomposition<T>,
          py::arg("a"), py::arg("result"), py::arg("power"), py::arg("base_bits"),
          "G decomposition (base 2^base_bits)");
}

template <typename T>
void bind_memory_ops(py::module_& m, const std::string& suffix) {
    m.def(("expand_" + suffix).c_str(),
          &expand<T>,
          py::arg("tensor"), py::arg("axis"), py::arg("repeats"),
          "Virtually expands the tensor along the given axis by repeating elements using stride tricks.");

    m.def(("squeeze_" + suffix).c_str(),
          &squeeze<T>,
          py::arg("tensor"), py::arg("axis"),
          "Removes a singleton dimension at the specified axis.");

    m.def(("unsqueeze_" + suffix).c_str(),
          &unsqueeze<T>,
          py::arg("tensor"), py::arg("axis"),
          "Inserts a singleton dimension at the specified axis.");
}

template <typename T>
void bind_device_memory(py::module_& m, const std::string& suffix) {
    using DeviceMem = DeviceTensor<T>;

    py::class_<DeviceMem, std::shared_ptr<DeviceMem>>(m, ("DeviceTensor" + suffix).c_str())
        .def("print", &DeviceMem::print)
        .def("print_metadata", &DeviceMem::print_metadata)
        .def("reshape", &DeviceMem::reshape);
}

template <typename T>
void bind_memory_helpers(py::module_& m, const std::string& suffix) {
    using namespace lattica_hw_api;
    m.def(("allocate_on_hardware_" + suffix).c_str(),
          &allocate_on_hardware<T>,
          py::arg("dims"));
    m.def(("host_to_device_" + suffix).c_str(),
          &host_to_device<T>,
          py::arg("tensor"));
    m.def(("device_to_host_" + suffix).c_str(),
          &device_to_host<T>,
          py::arg("device_mem"));
}

template <typename T>
void bind_contiguous(py::module_& m, const std::string& suffix) {
    m.def(("make_contiguous_" + suffix).c_str(), &make_contiguous<T>,
          py::arg("tensor"), "Return a contiguous version of the tensor.");
}

PYBIND11_MODULE(lattica_hw, m) {
    m.doc() = "Lattica Hardware API Python bindings";

    // Bind DeviceTensor class
    bind_device_memory<int32_t>(m, "32");
    bind_device_memory<int64_t>(m, "64");
    bind_device_memory<double>(m, "float64");

    // Bind memory ops
    bind_memory_helpers<int32_t>(m, "32");
    bind_memory_helpers<int64_t>(m, "64");
    bind_memory_helpers<double>(m, "float64");

    // Bind modular ops
    bind_modop_variants<int32_t>(m, "32");
    bind_modop_variants<int64_t>(m, "64");

    // axis_modsum
    m.def("axis_modsum_32", &axis_modsum<int32_t>, "Axis-wise modular sum (int32)");
    m.def("axis_modsum_64", &axis_modsum<int64_t>, "Axis-wise modular sum (int64)");

    // g_decomposition
    bind_g_decomposition<int32_t>(m, "32");
    bind_g_decomposition<int64_t>(m, "64");

    // bind expand, squeeze, unsqueeze
    bind_memory_ops<int32_t>(m, "32");
    bind_memory_ops<int64_t>(m, "64");
    bind_memory_ops<double>(m, "float64");

    // contiguous ops
    bind_contiguous<int32_t>(m, "32");
    bind_contiguous<int64_t>(m, "64");
    bind_contiguous<double>(m, "float64");

    // ntt
    m.def("ntt_32", &ntt<int32_t>, "NTT (int32)");
    m.def("ntt_64", &ntt<int64_t>, "NTT (int64)");

    // intt
    m.def("intt_32", &intt<int32_t>, "INTT (int32)");
    m.def("intt_64", &intt<int64_t>, "INTT (int64)");

    // permute
    m.def("permute_32", &permute<int32_t>, "Permute (int32)");
    m.def("permute_64", &permute<int64_t>, "Permute (int64)");

    m.def("pad_single_axis_32", &pad_single_axis<int32_t>, "Pad a single axis of a tensor (int32)");
    m.def("pad_single_axis_64", &pad_single_axis<int64_t>, "Pad a single axis of a tensor (int64)");
    m.def("pad_single_axis_float64", &pad_single_axis<double>, "Pad a single axis of a tensor (float64)");
}