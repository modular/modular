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

def DebugInfo_SubprogramScoped : OpInterface<
  "SubprogramScoped", [DebugInfo_ScopedLocation]
>{
  let description = [{
    This interface represents an operation that can have a debuginfo subprogram
    scope in its location, and a body whose operations (except other nested
    SubprogramScoped ops) must have the same scope.
  }];
  let cppNamespace = "::M::DebugInfo";

  let methods = [
    InterfaceMethod<[{
      Return the function-like region of the operation.
    }], "Region &", "getBodyRegion", (ins)>,
    InterfaceMethod<[{
      Return the location scope, or a null attribute if not available
    }], "DebugInfo::DISubprogramAttr", "getSubprogramScope", (ins), [{}],
    [{
      return dyn_cast_or_null<DebugInfo::DISubprogramAttr>(this->getLocScope());
    }]>,
  ];

  let extraTraitClassDeclaration = [{
    DebugInfo::DIScopeAttr getLocScope() {
      Location loc = this->getOperation()->getLoc();
      if (auto fusedLoc = dyn_cast<
          mlir::FusedLocWith<DebugInfo::DISubprogramAttr>>(loc))
        return fusedLoc.getMetadata();
      return {};
    }
  }];

  let verify = [{
    return impl::verifySubprogramScoped(cast<SubprogramScoped>($_op));
  }];
}

def DebugInfo_InlinedSubprogramScoped : OpInterface<
  "InlinedSubprogramScoped", [DebugInfo_SubprogramScoped]
>{
  let description = [{
    This interface represents a SubprogramScoped operation that also acts as the
    callsite to the subprogram represented by the op's body region. These ops
    have to carry additional location information of where the call is, which
    could contain callstack information due to inlining, etc.
  }];

  let dependentTraits = [DebugInfo_ScopedLocation];
  let cppNamespace = "::M::DebugInfo";

  let methods = [
    InterfaceMethod<[{
      Return the location of the call to the subprogram.
    }], "mlir::LocationAttr", "getCallLocAttr", (ins)>,
    InterfaceMethod<[{
      Set the location of the call to the subprogram.
    }], "void", "setCallLocAttr", (ins "mlir::LocationAttr":$attr)>,
  ];
}

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOINTERFACES_TD
