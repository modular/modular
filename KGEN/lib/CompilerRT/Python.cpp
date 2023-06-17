//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "llvm/ADT/StringRef.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Global PythonInterface Instance
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_Python_GetGlobalPython(ssize_t objSize,
                                       void (*initFn)(void *)) {
  static void *globalPython = nullptr;
  if (!globalPython) {
    globalPython = malloc(objSize);
    initFn(globalPython);
  }
  return globalPython;
}

//===----------------------------------------------------------------------===//
// CompilerRT Registration
//===----------------------------------------------------------------------===//

void KGEN::registerPython(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_Python_GetGlobalPython",
                   (void *)&KGEN_CompilerRT_Python_GetGlobalPython});
}
