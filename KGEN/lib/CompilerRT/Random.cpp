//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MathExtras.h"

extern "C" double KGEN_CompilerRT_RandomDouble(double min, double max) {
  double value;
  M::fillWithRandomFloats<double>(value, min, max);
  return value;
}

extern "C" int64_t KGEN_CompilerRT_RandomSInt64(int64_t min, int64_t max) {
  int64_t value;
  M::fillWithRandomInts<int64_t>(value, min, max);
  return value;
}

extern "C" uint64_t KGEN_CompilerRT_RandomUInt64(uint64_t min, uint64_t max) {
  uint64_t value;
  M::fillWithRandomInts<uint64_t>(value, min, max);
  return value;
}
