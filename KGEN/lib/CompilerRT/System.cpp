//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"
#include <stdio.h>
#include <thread>

/// Returns the number of cores on the system.
COMPILERRT_EXPORT size_t KGEN_CompilerRT_CoreCount() {
  return std::thread::hardware_concurrency();
}

void M::KGEN::registerSystem(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"fprintf", (void *)&fprintf});
  funcs.push_back(
      {"KGEN_CompilerRT_CoreCount", (void *)&KGEN_CompilerRT_CoreCount});
}
