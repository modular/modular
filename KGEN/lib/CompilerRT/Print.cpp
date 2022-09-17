//===- Print.cpp ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to print values.
//
//===----------------------------------------------------------------------===//

#include <cinttypes>
#include <cstdint>
#include <cstdio>

extern "C" {

/// Print an i32 value.
void printInt32(int32_t x) {
  printf("i32: %" PRId32 "\n", x);
  fflush(stdout);
}

/// Print an i64 value.
void printInt64(int64_t x) {
  printf("i64: %" PRId64 "\n", x);
  fflush(stdout);
}
}
