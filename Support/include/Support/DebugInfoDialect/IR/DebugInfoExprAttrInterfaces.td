//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef DEBUGINFOEXPRINTERFACES_TD
#define DEBUGINFOEXPRINTERFACES_TD

include "mlir/IR/Interfaces.td"

//===----------------------------------------------------------------------===//
// DIExprAttrInterface
//===----------------------------------------------------------------------===//

/// The base class of all debug info expression attributes. Expression
/// attributes are all typed attributes.
def DebugInfo_DIExprAttrInterface : AttrInterface<"DIExprAttr"> {
  let description = [{
    This interface is used for attributes that represent DebugInfo expression
    operators. It is a stronger version of TypedAttrInterface that enforces the
    type be a DIType.
  }];

  let cppNamespace = "::M::DebugInfo";
  let methods = [InterfaceMethod<
    "Get the attribute's DI type",
    "::M::DebugInfo::DIType", "getDIType"
  >];
}

#endif // DEBUGINFOEXPRINTERFACES_TD
