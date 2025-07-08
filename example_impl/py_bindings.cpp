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

    // mod variants
    m.def(("mod_tt_" + suffix).c_str(),
          &mod_tt<T>,
          py::arg("a"), py::arg("b"), py::arg("result"),
          "Elementwise modular remainder: ([...,k] % [...,k])");
    m.def(("mod_tc_" + suffix).c_str(),
          &mod_tc<T>,
          py::arg("a"), py::arg("b_scalar"), py::arg("result"),
          "Elementwise modular remainder: ([...,k] % scalar)");
    m.def(("mod_ct_" + suffix).c_str(),
          &mod_ct<T>,
          py::arg("a_scalar"), py::arg("b"), py::arg("result"),
          "Elementwise modular remainder: (scalar % [...,k])");

    // modneg variants
    m.def(("modneg_tt_" + suffix).c_str(),
          &modneg_tt<T>,
          py::arg("a"), py::arg("p"), py::arg("result"),
          "Elementwise modular negation: ([...,k] % [...,k])");
    m.def(("modneg_tc_" + suffix).c_str(),
          &modneg_tc<T>,
          py::arg("a"), py::arg("p_scalar"), py::arg("result"),
          "Elementwise modular negation: ([...,k] % scalar)");
}

template <typename T, typename U>
void bind_g_decomposition(py::module_& m, const std::string& suffix) {
    m.def(("apply_g_decomp_" + suffix).c_str(), &apply_g_decomp<T,U>,
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

    m.def(("moveaxis_" + suffix).c_str(),
          &moveaxis<T>,
          py::arg("tensor"), py::arg("axis_src"), py::arg("axis_dst"),
          "Moves an axis from axis_src to axis_dst.");

    m.def(("flatten_" + suffix).c_str(),
          &flatten<T>,
          py::arg("tensor"), py::arg("start_axis"), py::arg("end_axis"),
          "Flattens the tensor between start_axis and end_axis, inclusive.");

    m.def(("get_slice_" + suffix).c_str(),
          // [tensor, sliceList] → vector<SliceArg> → get_slice<T>
          [](const std::shared_ptr<DeviceTensor<T>>& tensor,
             py::iterable sliceList)
          {
              std::vector<SliceArg> args;
              args.reserve(std::distance(sliceList.begin(), sliceList.end()));

              for (py::handle h : sliceList) {
                  if (py::isinstance<py::int_>(h)) {
                      // integer → int64_t
                      args.emplace_back(h.cast<int64_t>());
                  }
                  else if (py::isinstance<py::slice>(h)) {
                      // Python slice → C++ Slice(start, stop, step)
                      py::slice sl = h.cast<py::slice>();
                      int64_t s  = sl.attr("start").cast<int64_t>();
                      int64_t e  = sl.attr("stop").cast<int64_t>();
                      int64_t st = sl.attr("step").cast<int64_t>();
                      args.emplace_back(Slice(s, e, st));
                  }
                  else {
                      throw std::invalid_argument(
                          "get_slice: sliceList must contain only ints or slices");
                  }
              }

              return get_slice<T>(tensor, args);
          },
          py::arg("tensor"),
          py::arg("sliceList"),
          R"doc(
            Returns a zero-copy view of `tensor` sliced along each axis.
            Accepts a Python iterable of ints and slice(start,stop,step),
            which are mapped into your C++ SliceArg variant.)doc");
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
    m.def(("empty_" + suffix).c_str(),
          &empty<T>,
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
    m.def(("contiguous_" + suffix).c_str(), &contiguous<T>,
          py::arg("tensor"), "Return a contiguous version of the tensor.");
}

PYBIND11_MODULE(lattica_hw, m) {
    m.doc() = "Lattica Hardware API Python bindings";

    // Bind DeviceTensor class
    bind_device_memory<int8_t>(m, "8");
    bind_device_memory<int32_t>(m, "32");
    bind_device_memory<int64_t>(m, "64");

    // Bind memory ops
    bind_memory_helpers<int8_t>(m, "8");
    bind_memory_helpers<int32_t>(m, "32");
    bind_memory_helpers<int64_t>(m, "64");
    m.def(("zeros_" + std::string("32")).c_str(), &zeros<int32_t>, py::arg("dims"));
    m.def(("zeros_" + std::string("64")).c_str(), &zeros<int64_t>, py::arg("dims"));

    // Bind modular ops
    bind_modop_variants<int32_t>(m, "32");
    bind_modop_variants<int64_t>(m, "64");

    // axis_modsum
    m.def("axis_modsum_32", &axis_modsum<int32_t>, "Axis-wise modular sum (int32)");
    m.def("axis_modsum_64", &axis_modsum<int64_t>, "Axis-wise modular sum (int64)");

    // modmul_axis_sum
    m.def("modmul_axis_sum_32", &modmul_axis_sum<int32_t>, "Element-wise modular multiply and sum over the specified axis (int32)");
    m.def("modmul_axis_sum_64", &modmul_axis_sum<int64_t>, "Element-wise modular multiply and sum over the specified axis (int64)");

    // g_decomposition
    bind_g_decomposition<int32_t, int8_t>(m, "32_8");
    bind_g_decomposition<int64_t, int8_t>(m, "64_8");
    bind_g_decomposition<int32_t, int32_t>(m, "32_32");
    bind_g_decomposition<int64_t, int64_t>(m, "64_64");

    // bind expand, squeeze, unsqueeze
    bind_memory_ops<int32_t>(m, "32");
    bind_memory_ops<int64_t>(m, "64");
    m.def(("moveaxis_" + std::string("8")).c_str(),
          &moveaxis<int8_t>,
          py::arg("tensor"), py::arg("axis_src"), py::arg("axis_dst"),
          "Moves an axis from axis_src to axis_dst.");
    m.def(("expand_" + std::string("8")).c_str(),
          &expand<int8_t>,
          py::arg("tensor"), py::arg("axis"), py::arg("repeats"),
          "Virtually expands the tensor along the given axis by repeating elements using stride tricks.");

    // contiguous ops
    bind_contiguous<int8_t>(m, "8");
    bind_contiguous<int32_t>(m, "32");
    bind_contiguous<int64_t>(m, "64");

    // ntt
    m.def("ntt_8_32", &ntt<int8_t, int32_t>, "NTT (int8)");
    m.def("ntt_8_64", &ntt<int8_t, int64_t>, "NTT (int8)");
    m.def("ntt_32_32", &ntt<int32_t, int32_t>, "NTT (int32)");
    m.def("ntt_64_64", &ntt<int64_t, int64_t>, "NTT (int64)");

    // intt
    m.def("intt_32", &intt<int32_t>, "INTT (int32)");
    m.def("intt_64", &intt<int64_t>, "INTT (int64)");

    // take_along_axis
    m.def("take_along_axis_32", &take_along_axis<int32_t>, py::arg("tensor"), py::arg("indices"), py::arg("axis"), py::arg("result"),
          "take_along_axis (int32)");
    m.def("take_along_axis_64", &take_along_axis<int64_t>, py::arg("tensor"), py::arg("indices"), py::arg("axis"), py::arg("result"),
          "take_along_axis (int64)");

    // set_const_val
    m.def("set_const_val_32", &set_const_val<int32_t>, py::arg("tensor"), py::arg("value"),
          "Set all elements of a tensor to a constant value (int32)");
    m.def("set_const_val_64", &set_const_val<int64_t>, py::arg("tensor"), py::arg("value"),
          "Set all elements of a tensor to a constant value (int64)");
}