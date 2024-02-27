//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./Memory.h"
#include "KGEN/CompilerRT/Registration.h"
#include "Support/AlignedAlloc.h"
#include "Support/Configuration.h"
#include "Support/SymbolExport.h"

using namespace M;

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_getConfigValue(const char *key) {
  ErrorOr<Config> configOr = Config::open();
  if (configOr.isError())
    return nullptr;

  StringRef value = configOr->getValue(key);
  char *res = (char *)KGEN_CompilerRT_AlignedAlloc(kPreferredMemoryAlignment,
                                                   value.size() + 1);
  strncpy(res, value.str().c_str(), value.size());
  res[value.size()] = 0;
  return res;
}

void M::KGEN::registerConfig(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.emplace_back(
      std::pair{llvm::StringLiteral("KGEN_CompilerRT_getConfigValue"),
                (void *)&KGEN_CompilerRT_getConfigValue});
}
