//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include <cstdarg>

extern "C" void KGEN_CompilerRT_PrintToStdErr(const char *data, ssize_t size) {
  llvm::errs() << llvm::StringRef(data, size);
  llvm::errs().flush();
}

extern "C" void KGEN_CompilerRT_PrintFormat(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stdout, fmt, args);
  fflush(stdout);
  va_end(args);
}
