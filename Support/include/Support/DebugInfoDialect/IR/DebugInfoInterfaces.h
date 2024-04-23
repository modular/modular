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
class InlinedSubprogramScoped;

/// Return true if constants should be materialized into a subprogram scoped
/// region.
bool shouldMaterializeConstantsInto(Region &region);

namespace impl {
LogicalResult verifySubprogramScoped(SubprogramScoped op);

Location getLocNoInlined(InlinedSubprogramScoped iss);
LocationAttr getCallLocAttr(InlinedSubprogramScoped iss);
void setCallLocAttr(InlinedSubprogramScoped iss, LocationAttr attr);
} // namespace impl
} // namespace M::DebugInfo

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h.inc"

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOINTERFACES_H
