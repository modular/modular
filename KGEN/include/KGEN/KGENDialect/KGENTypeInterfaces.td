//===----------------------------------------------------------------------===//
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
      "TypedAttr", "getDType">,
    InterfaceMethod<
      /*desc=*/[{
        Try to resolve the data type from the parameter. Returns
        `DType::invalid` if the type could not be resolved.
      }], "Optional<KGEN::KGENDType>", "getResolvedDType", (ins), [{}], [{
        if (auto constant = dyn_cast<DTypeConstantAttr>($_type.getDType()))
          return constant.getDType();
        return {};
      }]>
  ];
}

//===----------------------------------------------------------------------===//
// DataLayoutInterface
//===----------------------------------------------------------------------===//

def DataLayoutInterface : TypeInterface<"DataLayoutInterface"> {
  let description = [{
    This interface is used to query the size and layout of types.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the size in bytes of the type.",
      "Optional<int64_t>", "getTypeSize", (ins "TargetInfoAttr":$target)>,
    InterfaceMethod<
      /*desc=*/"Get the alignment in bytes of the type.",
      "Optional<int64_t>", "getTypeAlign", (ins "TargetInfoAttr":$target)>,
  ];

  let extraClassDeclaration = [{
    /// Try to get the size in bytes of the type.
    static Optional<int64_t>
    getTypeSizeInBytes(TargetInfoAttr target, Type type);

    /// Try to get the aligment in bytes of the type.
    static Optional<int64_t>
    getTypeAlignInBytes(TargetInfoAttr target, Type type);
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

/// A type whose data type conforms to a constraint.
class ConstrainedDType<string constraint> : ConfinedType<DTypeInterface,
    [CPred<!subst("$_self", "$_self.cast<DTypeInterface>().getResolvedDType()",
                 constraint)>]>;

/// A type whose data type is boolean.
def BoolDType : ConstrainedDType<"$_self && $_self->isBool()">;
/// A type whose data type is index.
def IndexDType : ConstrainedDType<"$_self && $_self->isIndex()">;
/// A type whose data type is integral.
def IntegerDType : ConstrainedDType<"$_self && $_self->isInt()">;
/// A type whose data type is floating point.
def FloatDType : ConstrainedDType<"$_self && $_self->isFloat()">;
/// A type whose data type is ui8.
def UI8DType : ConstrainedDType<"$_self && *$_self == KGENDType::ui8">;
/// A type whose data type is address.
def AddressDType : ConstrainedDType<"$_self && $_self->isAddress()">;
/// A type whose data type is either unbound.
def UnboundDType :
    ConstrainedDType<"!$_self">;
/// A type whose data type is either unbound or integral.
def IntegerOrUnboundDType :
    ConstrainedDType<"!$_self || $_self->isInt()">;
/// A type whose data type is either unbound or signed.
def UnboundOrSignedDType :
    ConstrainedDType<"!$_self || $_self->isSInt() || $_self->isFloat()">;
/// A type whose data type is either unbound or floating point.
def FloatOrUnboundDType :
    ConstrainedDType<"!$_self || $_self->isFloat()">;
/// A type whose data type is either index or arithmetic.
def IndexOrArithmeticDType :
    ConstrainedDType<"$_self && ($_self->isIndex() || $_self->isArithmetic())">;
/// A type whose data type is either unbound, index, or integral.
def UnboundOrIndexOrIntegerDType :
    ConstrainedDType<"!$_self || $_self->isIndex() || $_self->isInt()">;
/// A type whose data type is either unbound, index, or arithmetic.
def UnboundOrIndexOrArithmeticDType :
    ConstrainedDType<"!$_self || $_self->isIndex() || $_self->isArithmetic()">;
/// A type whose data type is either unbound, index, integral, or bool.
def IndexOrIntegerOrBoolOrUnboundDType :
    ConstrainedDType<
      "!$_self || $_self->isIndex() || $_self->isInt() || $_self->isBool()">;

#endif // KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
