# Specify the library source files
set(SOURCES
    core_impl.cpp
    modsum_impl.cpp
    modmul_impl.cpp
    axis_modsum_impl.cpp
    g_decomposition_impl.cpp
    ntt_impl.cpp
    permute_impl.cpp
)

# Create a static library for the Example Implementation
add_library(example_impl STATIC ${SOURCES})

# Add the include directory for this library
target_include_directories(example_impl PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
