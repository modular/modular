//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/AlignedAlloc.h"
#include "Support/Configuration.h"
#include "Support/SymbolExport.h"

using namespace M;

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_getConfigValue(const char *key) {
  llvm::StringRef value;
  ErrorOr<Config> configOr = Config::open();
  if (configOr.isError())
    value = "";

  Config cfg = std::move(*configOr);

  // getValue may return empty string if key does not exist,
  // in which case we finally return an empty string.
  value = cfg.getValue(key);
  char *cvalue = static_cast<char *>(
      alignedAlloc(kPreferredMemoryAlignment, value.size() + 1));
  strcpy(cvalue, value.str().c_str());
  return cvalue;
}
