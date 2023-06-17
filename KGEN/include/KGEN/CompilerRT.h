//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_RT_H
#define KGEN_COMPILER_RT_H

#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"

/// This file includes at least one declaration from each .cpp file in the
/// CompilerRT directory. This is used to ensure that the functions defined are
/// marked as 'used' in a linked executable, and can be accessed by the JIT in
/// that process.

//===----------------------------------------------------------------------===//
// Initialize.cpp
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED bool KGEN_CompilerRT_Initialize();

//===----------------------------------------------------------------------===//
// InitIntelAMX.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the intel AMX functions.
void registerIntelAMX(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
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

namespace M::KGEN {
/// Register the Python functions.
void registerPython(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

#endif // KGEN_COMPILER_RT_H
