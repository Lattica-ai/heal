# Specify the library source files
set(SOURCES
    device_tensor_ex.cpp
    device_memory.cpp
    modular_ops.cpp
    modular_axis_sum_ops.cpp
    gadget_decomposition.cpp
    ntt.cpp
    tensor_layout_ops.cpp
    tensor_value_ops.cpp
)

# Create a static library for the Example Implementation
add_library(example_impl STATIC ${SOURCES})

# Ensure position-independent code (PIC) for use in shared libs
set_target_properties(example_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Add the include directory for this library
target_include_directories(example_impl PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Optional -- use OpenMP for parallelization
find_package(OpenMP REQUIRED)
target_link_libraries(example_impl PUBLIC OpenMP::OpenMP_CXX)


# Find pybind11
find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(lattica_hw py_bindings.cpp)

# Link against example_impl
target_link_libraries(lattica_hw PRIVATE example_impl torch_python)
target_include_directories(lattica_hw PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
