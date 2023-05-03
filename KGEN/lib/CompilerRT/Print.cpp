//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "Support/SymbolExport.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdarg>

COMPILERRT_EXPORT void KGEN_CompilerRT_PrintToStdErr(const char *data,
                                                     ssize_t size) {
  llvm::errs() << llvm::StringRef(data, size);
  llvm::errs().flush();
}

// TODO(#9034): Why do we need this?
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_PrintFormat(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stdout, fmt, args);
  fflush(stdout);
  va_end(args);
}

void M::KGEN::registerPrint(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_PrintToStdErr",
                   (void *)&KGEN_CompilerRT_PrintToStdErr});
  funcs.push_back(
      {"KGEN_CompilerRT_PrintFormat", (void *)&KGEN_CompilerRT_PrintFormat});
}
