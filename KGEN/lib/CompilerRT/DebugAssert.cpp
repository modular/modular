//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"

#include "KGEN/CompilerRT.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstdio>

MODULAR_EXPORT void KGEN_CompilerRT_DebugAssert(bool cond, const char *funcName,
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

void M::KGEN::registerDebugAssert(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back(
      {"KGEN_CompilerRT_DebugAssert", (void *)&KGEN_CompilerRT_DebugAssert});
}
