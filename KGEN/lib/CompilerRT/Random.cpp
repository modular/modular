//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MathExtras.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetRandomState() {
  static std::default_random_engine randEngine(/*seed=*/0);
  return &randEngine;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_SetRandomStateSeed(std::default_random_engine *engine,
                                   ssize_t seed) {
  engine->seed(seed);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT double
KGEN_CompilerRT_RandomDouble(double min, double max) {
  std::uniform_real_distribution<double> dist(min, max);
  return dist(*(std::default_random_engine *)KGEN_CompilerRT_GetRandomState());
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT int64_t
KGEN_CompilerRT_RandomSInt64(int64_t min, int64_t max) {
  std::uniform_int_distribution<int64_t> dist(min, max);
  return dist(*(std::default_random_engine *)KGEN_CompilerRT_GetRandomState());
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint64_t
KGEN_CompilerRT_RandomUInt64(uint64_t min, uint64_t max) {
  std::uniform_int_distribution<uint64_t> dist(min, max);
  return dist(*(std::default_random_engine *)KGEN_CompilerRT_GetRandomState());
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT double
KGEN_CompilerRT_NormalDouble(double mean, double standardDeviation) {
  std::normal_distribution<double> dist{mean, standardDeviation};
  return dist(*(std::default_random_engine *)KGEN_CompilerRT_GetRandomState());
}
