//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_TRANSFORMS_PASSES_H
#define SUPPORT_DEBUGINFODIALECT_TRANSFORMS_PASSES_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"

#include "mlir/Pass/Pass.h"
#include "llvm/BinaryFormat/Dwarf.h"

namespace M::DebugInfo {
//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Support/DebugInfoDialect/Transforms/Transforms.h.inc"

} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_TRANSFORMS_PASSES_H
