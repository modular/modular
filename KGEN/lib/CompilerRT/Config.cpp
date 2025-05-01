//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Memory.h"
#include "Support/AlignedAlloc.h"
#include "Support/Configuration.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/Twine.h"

using namespace M;

#define _STRINGIFY(str) #str
#define _X_STRINGIFY(str) _STRINGIFY(str)
#define STRINGIFY_MAX_CONFIG _X_STRINGIFY(MAX_CONFIG_SECTION)

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_getMAXConfigValue(const char *key, size_t length) {
  ErrorOr<Config> configOr = Config::open();
  if (configOr.isError())
    return nullptr;

  std::string configKey =
      (llvm::Twine(STRINGIFY_MAX_CONFIG).concat(StringRef(key, length))).str();
  StringRef value = configOr->getValue(configKey);
  char *res = (char *)KGEN_CompilerRT_AlignedAlloc(kPreferredMemoryAlignment,
                                                   value.size() + 1);
  strncpy(res, value.str().c_str(), value.size());
  res[value.size()] = 0;
  return res;
}
