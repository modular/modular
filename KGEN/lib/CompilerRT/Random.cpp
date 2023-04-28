//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "Support/MathExtras.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

COMPILERRT_EXPORT double KGEN_CompilerRT_RandomDouble(double min, double max) {
  static std::default_random_engine randEngine(/*seed=*/0);
  std::uniform_real_distribution<double> dist(min, max);
  return dist(randEngine);
}

COMPILERRT_EXPORT int64_t KGEN_CompilerRT_RandomSInt64(int64_t min,
                                                       int64_t max) {
  static std::default_random_engine randEngine(/*seed=*/0);
  std::uniform_int_distribution<int64_t> dist(min, max);
  return dist(randEngine);
}

COMPILERRT_EXPORT uint64_t KGEN_CompilerRT_RandomUInt64(uint64_t min,
                                                        uint64_t max) {
  static std::default_random_engine randEngine(/*seed=*/0);
  std::uniform_int_distribution<uint64_t> dist(min, max);
  return dist(randEngine);
}

void M::KGEN::registerRandom(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back(
      {"KGEN_CompilerRT_RandomDouble", (void *)&KGEN_CompilerRT_RandomDouble});
  funcs.push_back(
      {"KGEN_CompilerRT_RandomSInt64", (void *)&KGEN_CompilerRT_RandomSInt64});
  funcs.push_back(
      {"KGEN_CompilerRT_RandomUInt64", (void *)&KGEN_CompilerRT_RandomUInt64});
}
