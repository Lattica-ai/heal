# Specify the library source files
set(SOURCES
    device_memory_impl.cpp
    modop_impl.cpp
    axis_modsum_impl.cpp
    g_decomposition_impl.cpp
    ntt_impl.cpp
    permute_impl.cpp
    memory_virtual_ops_impl.cpp
    contiguous_impl.cpp
    compute_ops_impl.cpp
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
