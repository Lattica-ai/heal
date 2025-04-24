// tests/test_helpers.h
#pragma once

#include <gtest/gtest.h>

#ifdef ENABLE_VALIDATION_TESTS
# define VALIDATION_RUN_TEST() true
#else
# define VALIDATION_RUN_TEST() false
#endif

#define VALIDATION_TEST(suite, name)                                           \
  /* forward-declare your body */                                              \
  void suite##_##name##_ValidationBody();                                      \
                                                                               \
  /* the real test; skips or calls into your ValidationBody */                 \
  TEST(suite, name) {                                                          \
    if (!VALIDATION_RUN_TEST()) {                                              \
      GTEST_SKIP() << "Skipping validation test: " #suite "." #name;           \
      return;                                                                  \
    }                                                                          \
    suite##_##name##_ValidationBody();                                         \
  }                                                                            \
                                                                               \
  void suite##_##name##_ValidationBody()
