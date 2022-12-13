//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRINTERFACES_TD
#define KGEN_KGENDIALECT_KGENATTRINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ParameterAttr
//===----------------------------------------------------------------------===//

def ParameterAttrInterface : AttrInterface<"ParameterAttr"> {
  let description = [{
    Any attribute that implements `TypedAttr` can be used as a parameter
    attribute in KGEN, but this interface allows parameter attributes to plug
    into specific parts of the KGEN parameter system.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Returns whether the attribute can be considered a simple constant.
    }], "bool", "isSimpleConstant">,
    InterfaceMethod<[{
      Compare this parameter attribute with another attribute that may not be of
      the same kind, returning true if this parameter should be considered
      "less" than the other attribute.
    }], "bool", "compare", (ins "Attribute":$rhs)>,
  ];

  let extraClassDeclaration = [{
    /// Return true if the parameter can be considered a "simple" constant. A
    /// parameter is a constant if it contains no nested expressions that have
    /// not been evaluated.
    static bool isSimpleConstant(Attribute attr);
  }];
}

#endif // KGEN_KGENDIALECT_KGENATTRINTERFACES_TD
