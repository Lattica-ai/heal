# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(LatticaHardwareAPI VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the example_impl subdirectory
add_subdirectory(example_impl)

# Add the tests directory and enable_testing
add_subdirectory(tests)
enable_testing()