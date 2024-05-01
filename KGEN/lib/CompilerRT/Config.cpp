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
#include "llvm/ADT/Twine.h"

using namespace M;

#define _STRINGIFY(str) #str
#define _X_STRINGIFY(str) _STRINGIFY(str)
#define STRINGIFY_MAX_CONFIG _X_STRINGIFY(MAX_CONFIG_SECTION)

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_getMAXConfigValue(const char *key) {
  ErrorOr<Config> configOr = Config::open();
  if (configOr.isError())
    return nullptr;

  llvm::Twine configKey = llvm::Twine(STRINGIFY_MAX_CONFIG).concat(key);
  StringRef value = configOr->getValue(configKey.str());
  char *res = (char *)KGEN_CompilerRT_AlignedAlloc(kPreferredMemoryAlignment,
                                                   value.size() + 1);
  strncpy(res, value.str().c_str(), value.size());
  res[value.size()] = 0;
  return res;
}

void M::KGEN::registerConfig(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.emplace_back(
      std::pair{llvm::StringLiteral("KGEN_CompilerRT_getMAXConfigValue"),
                (void *)&KGEN_CompilerRT_getMAXConfigValue});
}
