//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides support for stripping debug information from IR.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_TRANSFORMS_STRIPDEBUGINFO_H
#define SUPPORT_DEBUGINFODIALECT_TRANSFORMS_STRIPDEBUGINFO_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::DebugInfo {

/// Parse a source file contained within the given source manager, and attach
/// artificial debug information that describes the input IR.
void stripDebugInfo(Operation *scope, bool preserveLineTables);

} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_TRANSFORMS_STRIPDEBUGINFO_H
