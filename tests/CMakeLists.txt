# Enable testing
enable_testing()

# Fetch GoogleTest package
include(FetchContent)
# Pull GoogleTest (release-1.12.1 is stable and safe)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# For MSVC builds; safe to leave in
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Make it available for subdirectories
FetchContent_MakeAvailable(googletest)

# Include directories
include_directories(${GTEST_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

# Define the list of test sources and their corresponding test names
set(TEST_SOURCES
    test_ntt.cpp
    test_modular_ops.cpp
    test_modular_axis_sum_ops.cpp
    test_gadget_decomposition_full_q.cpp
    test_decomp_reconstruct.cpp
    test_tensor_layout_ops.cpp
    test_contiguous.cpp
    test_tensor_value_ops.cpp
)

set(TEST_NAMES
    NTTTests
    ModularOpsTests
    ModularAxisSumOpsTests
    TensorGadgetDecompositionTests
    GadgetDecompositionFullQTests
    DecompReconstructTests
    TensorLayoutOpsTests
    ContiguousTests
    TensorValueOpsTests
)

# Loop through the test sources and add executables and tests
list(LENGTH TEST_SOURCES NUM_TESTS)
math(EXPR NUM_TESTS "${NUM_TESTS} - 1")
foreach(i RANGE 0 ${NUM_TESTS})
    list(GET TEST_SOURCES ${i} TEST_SOURCE)
    list(GET TEST_NAMES ${i} TEST_NAME)

    # Create the executable for the test
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} example_impl GTest::gtest GTest::gtest_main)

    # Add the test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
