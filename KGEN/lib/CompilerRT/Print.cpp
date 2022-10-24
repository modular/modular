//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"

extern "C" void KGEN_CompilerRT_PrintToStdErr(const char *data, ssize_t size) {
  llvm::errs() << llvm::StringRef(data, size);
  llvm::errs().flush();
}
