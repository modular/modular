//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef DEBUGINFOEXPRINTERFACES_TD
#define DEBUGINFOEXPRINTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// DIExprAttrInterface
//===----------------------------------------------------------------------===//

/// The base class of all debug info expression attributes. Expression
/// attributes are all typed attributes.
def DebugInfo_DIExprAttrInterface
  : AttrInterface<"DIExprAttr", [TypedAttrInterface]> {
  let description = [{
    This interface is used for attributes that represent DebugInfo expression
    operators.

    Each operator is typed so that dialects may type-check DI expressions.
    However, since neither LLVM nor DWARF expects typed DI expressions, these
    types will only be used for legalization during compilation, and are
    discarded during LLVM lowering. Thus, the type of an expression operator is
    not expected to be a DIType. This also allows dialects to use their native
    types to track the optimizations their passes performed on them.
  }];
  let cppNamespace = "::M::DebugInfo";
}

#endif // DEBUGINFOEXPRINTERFACES_TD
