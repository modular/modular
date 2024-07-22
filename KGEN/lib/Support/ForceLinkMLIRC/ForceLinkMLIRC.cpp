//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"

#include "AffineExpr.cpp.inc"
#include "AffineMap.cpp.inc"
#include "BuiltinAttributes.cpp.inc"
#include "BuiltinTypes.cpp.inc"
#include "Diagnostics.cpp.inc"
#include "IR.cpp.inc"
#include "IntegerSet.cpp.inc"
#include "Pass.cpp.inc"
#include "Rewrite.cpp.inc"
#include "Support.cpp.inc"
#include "Transforms.cpp.inc"

namespace M::KGEN {

/// Calling this function forces the linking of MLIR C API symbols. This allows
/// JIT'ed Mojo code to use the same MLIR C API symbols as the current process,
/// which is necessary to avoid conflicting TypeIDs.
void forceLinkMLIRC() {
  forceLinkMLIRCAffineExpr();
  forceLinkMLIRCAffineMap();
  forceLinkMLIRCBuiltinAttributes();
  forceLinkMLIRCBuiltinTypes();
  forceLinkMLIRCDiagnostics();
  forceLinkMLIRCIR();
  forceLinkMLIRCIntegerSet();
  forceLinkMLIRCPass();
  forceLinkMLIRCRewrite();
  forceLinkMLIRCSupport();
  forceLinkMLIRCTransforms();
}

} // namespace M::KGEN
