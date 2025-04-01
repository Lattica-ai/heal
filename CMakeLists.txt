# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(LatticaHardwareAPI VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable or disable based on whether the torch version in the Python virtual environment is pre-cxx11
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# link to the libtorch version that is installed in the Python virtual environment
# instead of the one that is installed in the system
#set(VENV_PATH "$ENV{VIRTUAL_ENV}")
#message(STATUS "VENV_PATH: ${VENV_PATH}")
#set(TORCH_LIB_DIR "${VENV_PATH}/lib/python3.12/site-packages/torch/lib")
#set(TORCH_INCLUDE_DIR "${VENV_PATH}/lib/python3.12/site-packages/torch/include")
#set(TORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIR};${TORCH_INCLUDE_DIR}/torch/csrc/api/include")

# Get torch.__path__[0]
execute_process(
    COMMAND python3 -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_SITE_PACKAGES_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set include and lib directories based on that path
set(TORCH_INCLUDE_DIR "${TORCH_SITE_PACKAGES_PATH}/include")
set(TORCH_LIB_DIR "${TORCH_SITE_PACKAGES_PATH}/lib")
set(TORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIR};${TORCH_INCLUDE_DIR}/torch/csrc/api/include")

message(STATUS "Resolved TORCH_INCLUDE_DIR: ${TORCH_INCLUDE_DIR}")
message(STATUS "Resolved TORCH_LIB_DIR: ${TORCH_LIB_DIR}")

include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIB_DIR})
link_libraries(torch torch_cpu c10)


# Add the example_impl subdirectory
add_subdirectory(example_impl)

# Add the tests directory and enable_testing
add_subdirectory(tests)
enable_testing()