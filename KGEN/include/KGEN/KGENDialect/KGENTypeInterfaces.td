//===- KGENTypeInterfaces.td ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
#define KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DTypeInterface
//===----------------------------------------------------------------------===//

def DTypeInterface : TypeInterface<"DTypeInterface"> {
  let description = [{
    This interface is used to interact with types that contain a parametric
    data type parameter.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the typed expression representing the data type.",
      "TypedAttr", "getDType">
  ];

  let extraSharedClassDeclaration = [{
    /// Try to resolve the data type from the parameter. Returns
    /// `DType::invalid` if the type could not be resolved.
    DType resolveDType();
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

/// A type whose data type conforms to a constraint.
class ConstrainedDType<string constraint> : ConfinedType<DTypeInterface,
    [CPred<!subst("$_self", "$_self.cast<DTypeInterface>().resolveDType()",
                 constraint)>]>;

/// A type whose data type is boolean.
def BoolDType : ConstrainedDType<"$_self.isBool()">;
/// A type whose data type is either unboard or floating point.
def FloatOrUnboundDType :
    ConstrainedDType<"$_self.isInvalid() || $_self.isFloat()">;

#endif // KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
