//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
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
KGEN_CompilerRT_RandomDouble(std::default_random_engine *engine, double min,
                             double max) {
  std::uniform_real_distribution<double> dist(min, max);
  return dist(*engine);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT int64_t
KGEN_CompilerRT_RandomSInt64(std::default_random_engine *engine, int64_t min,
                             int64_t max) {
  std::uniform_int_distribution<int64_t> dist(min, max);
  return dist(*engine);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint64_t
KGEN_CompilerRT_RandomUInt64(std::default_random_engine *engine, uint64_t min,
                             uint64_t max) {
  std::uniform_int_distribution<uint64_t> dist(min, max);
  return dist(*engine);
}

void M::KGEN::registerRandom(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_GetRandomState",
                   (void *)&KGEN_CompilerRT_GetRandomState});
  funcs.push_back({"KGEN_CompilerRT_SetRandomStateSeed",
                   (void *)&KGEN_CompilerRT_SetRandomStateSeed});
  funcs.push_back(
      {"KGEN_CompilerRT_RandomDouble", (void *)&KGEN_CompilerRT_RandomDouble});
  funcs.push_back(
      {"KGEN_CompilerRT_RandomSInt64", (void *)&KGEN_CompilerRT_RandomSInt64});
  funcs.push_back(
      {"KGEN_CompilerRT_RandomUInt64", (void *)&KGEN_CompilerRT_RandomUInt64});
}
