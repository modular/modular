//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"
#include <cstdarg>
#include <stdio.h>
#include <thread>

/// Returns the number of cores on the system.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_CoreCount() {
  return std::thread::hardware_concurrency();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT int
KGEN_CompilerRT_fprintf(FILE *stream, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int result = vfprintf(stream, format, args);
  va_end(args);
  return result;
}

void M::KGEN::registerSystem(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back(
      {"KGEN_CompilerRT_fprintf", (void *)&KGEN_CompilerRT_fprintf});
  funcs.push_back(
      {"KGEN_CompilerRT_CoreCount", (void *)&KGEN_CompilerRT_CoreCount});
}
