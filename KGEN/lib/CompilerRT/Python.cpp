//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "llvm/ADT/StringRef.h"

using namespace M;

namespace {
// Must match the layout of PythonVersion in Kernels/mojo/Python/CPython.mojo
struct PythonVersion {
  ssize_t major = 0; // Int
  ssize_t minor = 0; // Int
  ssize_t patch = 0; // Int
};

// Must match the layout of CPython in Kernels/mojo/Python/CPython.mojo
struct CPython {
  void *lib = nullptr;              // DLHandle
  void *noneType = nullptr;         // PyObjectPtr
  void *dictType = nullptr;         // PyObjectPtr
  char loggingEnabled = false;      // Bool
  PythonVersion version{};          // PythonVersion
  ssize_t *totalRefCount = nullptr; // Pointer[Int]
};
} // namespace

//===----------------------------------------------------------------------===//
// Global PythonInterface Instance
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_Python_GetGlobalPython(ssize_t objSize,
                                       void (*initFn)(void *)) {
  static CPython globalPython{};
  if (!globalPython.lib)
    initFn(&globalPython);
  return &globalPython;
}

//===----------------------------------------------------------------------===//
// CompilerRT Registration
//===----------------------------------------------------------------------===//

void KGEN::registerPython(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_Python_GetGlobalPython",
                   (void *)&KGEN_CompilerRT_Python_GetGlobalPython});
}
