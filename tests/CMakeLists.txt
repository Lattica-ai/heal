# Enable testing
enable_testing()

# Find GoogleTest package
find_package(GTest REQUIRED)

# Include directories
include_directories(${GTEST_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

# Define the list of test sources and their corresponding test names
set(TEST_SOURCES
    test_ntt.cpp
    test_permute.cpp
    test_modop_v1.cpp
    test_modop_v2.cpp
    test_modop_v3.cpp
    test_axis_modsum.cpp
    test_g_decomposition.cpp
    test_decomp_reconstruct.cpp
    test_reshape.cpp
)

set(TEST_NAMES
    NTTTests
    PermuteTests
    ModOpV1Tests
    ModOpV2Tests
    ModOpV3Tests
    AxisModSumTests
    GDecompositionTests
    DecompReconstructTests
    ReshapeTests
)

# Loop through the test sources and add executables and tests
list(LENGTH TEST_SOURCES NUM_TESTS)
math(EXPR NUM_TESTS "${NUM_TESTS} - 1")
foreach(i RANGE 0 ${NUM_TESTS})
    list(GET TEST_SOURCES ${i} TEST_SOURCE)
    list(GET TEST_NAMES ${i} TEST_NAME)

    # Create the executable for the test
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} example_impl GTest::GTest GTest::Main)

    # Add the test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
