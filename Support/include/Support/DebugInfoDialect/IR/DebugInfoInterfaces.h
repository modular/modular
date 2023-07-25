//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOINTERFACES_H
#define SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOINTERFACES_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "mlir/IR/OpDefinition.h"

namespace M::DebugInfo {
class SubprogramScoped;

namespace impl {
LogicalResult verifySubprogramScoped(SubprogramScoped op);
} // namespace impl
} // namespace M::DebugInfo

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h.inc"

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOINTERFACES_H
