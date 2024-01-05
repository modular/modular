//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DIEXPRESSIONSIMPLIFIER_H
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DIEXPRESSIONSIMPLIFIER_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::DebugInfo {
/// Register all the known patterns for simplifying LLVM DIExpressions and apply
/// to all DebugValue & DebugDeclareOps in op.
void simplifyLLVMDIExpressionRecursively(Operation *op);
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DIEXPRESSIONSIMPLIFIER_H
