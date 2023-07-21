//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOINTERFACES_TD
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOINTERFACES_TD

include "mlir/IR/OpBase.td"

def DebugInfo_ScopedLocation : OpInterface<"ScopedLocation"> {
  let description = [{
    This interface represents an operation that can have a debuginfo scope in
    its location, and that scope can be retrieved quickly without recursing into
    nested locations.
  }];
  let cppNamespace = "::M::DebugInfo";

  let methods = [
    InterfaceMethod<[{
        Return the location scope, or a null attribute if not available
      }], "DebugInfo::DIScopeAttr", "getLocScope", (ins)>,
  ];
}

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOINTERFACES_TD
