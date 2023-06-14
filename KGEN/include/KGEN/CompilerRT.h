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

#if defined(__x86_64__) && defined(__linux__)
COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED bool KGEN_CompilerRT_Init_Intel_AMX();
#endif

//===----------------------------------------------------------------------===//
// LLCL.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the LLCL functions.
void registerLLCL(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED void KGEN_CompilerRT_LLCL_Dummy();

//===----------------------------------------------------------------------===//
// Memory.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the Memory functions.
void registerMemory(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED void *
KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size);

//===----------------------------------------------------------------------===//
// Random.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the Random functions.
void registerRandom(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED double
KGEN_CompilerRT_RandomDouble(double min, double max);

//===----------------------------------------------------------------------===//
// System.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the system functions.
void registerSystem(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED size_t KGEN_CompilerRT_CoreCount();

//===----------------------------------------------------------------------===//
// Tracing.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the Tracing functions.
void registerTracing(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED void
KGEN_CompilerRT_TimeTraceProfilerEnd();

//===----------------------------------------------------------------------===//
// Python.cpp
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Register the Python functions.
void registerPython(std::vector<std::pair<llvm::StringLiteral, void *>> &funcs);
} // namespace M::KGEN

COMPILERRT_EXPORT LLVM_ATTRIBUTE_USED void *
KGEN_CompilerRT_Python_GetGlobalPython(ssize_t objSize, void (*initFn)(void *));

//===----------------------------------------------------------------------===//
// Linkage
//===----------------------------------------------------------------------===//

/// This declaration is used to ensure that the individual .o files are linked
/// into things that include this header. We only need to 'call' one function
/// from each .cpp file. Note that this function should never actually be
/// called!
MODULAR_VISIBILITY_EXPORT LLVM_ATTRIBUTE_USED inline void
KGEN_CompilerRT_dummylinkageinit() {
  KGEN_CompilerRT_Initialize();
#if defined(__x86_64__) && defined(__linux__)
  KGEN_CompilerRT_Init_Intel_AMX();
#endif
  KGEN_CompilerRT_LLCL_Dummy();
  KGEN_CompilerRT_RandomDouble(0, 0);
  KGEN_CompilerRT_CoreCount();
  KGEN_CompilerRT_TimeTraceProfilerEnd();
  KGEN_CompilerRT_Python_GetGlobalPython(0, nullptr);
}

#endif // KGEN_COMPILER_RT_H
