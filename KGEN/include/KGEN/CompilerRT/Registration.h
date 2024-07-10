//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILERRT_REGISTRATION_H
#define KGEN_COMPILERRT_REGISTRATION_H

#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include <vector>

/// This file includes at least one declaration from each .cpp file in the
/// CompilerRT directory. This is used to ensure that the functions defined are
/// marked as 'used' in a linked executable, and can be accessed by the JIT in
/// that process.

//===----------------------------------------------------------------------===//
// Initialize.cpp
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED bool KGEN_CompilerRT_Initialize();

//===----------------------------------------------------------------------===//
// Config.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the config handling functions.
void registerConfig(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Globals.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the global handling functions.
void registerGlobals(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

/// Allow parts of the execution engine to inject globals.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_InsertGlobal(llvm::StringRef name, void *value);

//===----------------------------------------------------------------------===//
// InitIntelAMX.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the intel AMX functions.
void registerIntelAMX(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// IO.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the IO functions.
void registerIO(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// LLCL.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the LLCL functions.
void registerLLCL(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Memory.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the Memory functions.
void registerMemory(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Random.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the Random functions.
void registerRandom(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// System.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the support functions.
void registerSupport(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// System.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the system functions.
void registerSystem(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Tracing.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the Tracing functions.
void registerTracing(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Python.cpp
//===----------------------------------------------------------------------===//

/// If not already set, this sets the `PYTHONPATH` environment variable to point
/// to the typical directories that contain Python modules. These directories
/// are discovered by invoking `python` and querying it for the paths it has
/// been configured to use.
///
/// If an error prevents `PYTHONPATH` from being set, this returns a pointer to
/// a non-empty string literal with an error message. Otherwise, this returns a
/// pointer to an empty string literal.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_Python_SetPythonPath();

namespace M::KGEN {
/// Register the Python functions.
void registerPython(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

#endif // KGEN_COMPILERRT_REGISTRATION_H
