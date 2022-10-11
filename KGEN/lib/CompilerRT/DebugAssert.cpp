//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"

#include <cassert>
#include <cstdio>

extern "C" void KGEN_CompilerRT_DebugAssert(bool cond, const char *funcName,
                                            const char *fileName,
                                            const char *message) {
  if (cond)
    return;
  fprintf(stderr, "%s:%s: failed assertion", fileName, funcName);
  if (message)
    fprintf(stderr, " '%s'", message);
  fprintf(stderr, "\n");
  fflush(stderr);
  LLVM_BUILTIN_TRAP;
  LLVM_BUILTIN_UNREACHABLE;
}
