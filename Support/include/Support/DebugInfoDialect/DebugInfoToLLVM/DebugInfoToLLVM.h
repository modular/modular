//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DEBUGINFOTOLLVM_H
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DEBUGINFOTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
#define GEN_PASS_DECL_DEBUGINFOTOLLVM
#define GEN_PASS_REGISTRATION
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h.inc"
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_DEBUGINFOTOLLVM_H
