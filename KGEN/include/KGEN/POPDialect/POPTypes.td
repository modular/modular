//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPTYPES_TD
#define KGEN_POPDIALECT_POPTYPES_TD

include "KGEN/KGENDialect/KGENTypeInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/POPDialect/POPDialect.td"
include "Support/MDialect/MTypeInterfaces.td"
include "KGEN/Interpreter/InterpreterInterface.td"
include "mlir/IR/AttrTypeBase.td"

class POPType<string typeName, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<POPDialect, typeName,
            [DeclareTypeInterfaceMethods<DataLayoutInterface>] # traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def POP_ArrayType : POPType<"Array", "array", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized array type";
  let description = [{
    This type represents a statically-sized array of elements. The size and
    element type are parameterized.

    Example:

    ```mlir
    // An array of four scalar floats.
    !pop.array<4, scalar<f32>>

    // A parameterized array.
    !pop.array<size, type>
    ```
  }];

  let parameters = (ins "TypedAttr":$size, "Type":$elementType);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` ` `
        custom<ParamType>($elementType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size,
                                        "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "int64_t":$size, "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "ValueRange":$elements)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the size expression to an integer.
    std::optional<int64_t> getResolvedSize() const;
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class ArrayElementType<string lhs, string rhs> : TypesMatchWith<
    "value type matches array element type", lhs, rhs,
    "llvm::cast<POP::ArrayType>($_self).getElementType()",
    "typeRangeMatches">;

def ArrayKnownSize : ConfinedType<POP_ArrayType,
    [CPred<"llvm::cast<POP::ArrayType>($_self).getResolvedSize()">],
    "array with known size">;

//===----------------------------------------------------------------------===//
// UnionType
//===----------------------------------------------------------------------===//

def POP_UnionType : POPType<"Union", "union", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "a parameterized union type";
  let description = [{
    The `!pop.union` type is a type that can contain any one of its union
    element types. Its size is large enough to contain the largest element and
    its alignment meets the strictest of its elements' alignment requirements.
    Unlike `!kgen.variant`, this union is not discriminated.

    Example:

    ```mlir
    !kgen.union<>
    !kgen.union<i32>
    !kgen.union<i32, i64>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"Type">:$types);
  let assemblyFormat = "`<` (`>`) : (custom<ParamTypes>($types)^ `>`)?";
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

def POP_SIMDType : POPType<"SIMD", "simd", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized SIMD vector type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.
  }];

  let parameters = (ins "TypedAttr":$size, "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` ` `
        custom<DTypeParamValue>($dType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size, "TypedAttr":$dtype)>,
    TypeBuilderWithInferredContext<(ins "int64_t":$size, "TypedAttr":$dtype)>,
    TypeBuilder<(ins "int64_t":$size, "KGENDType":$dtype)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the dtype expression of the SIMD type to a value.
    std::optional<KGENDType> getResolvedDType() const;
    /// Try to narrow the size expression of the SIMD type to an integer.
    std::optional<int64_t> getResolvedSize() const;
    /// Returns true if this is known to be a scalar type.
    bool isScalar() const { return getResolvedSize() == 1; }
  }];
}

//===----------------------------------------------------------------------===//
// TypeConstraints

/// A DType constraint.
class DTypeConstraint<string value> {
  string pred = value;
}

class AndDType<list<DTypeConstraint> preds> :
    DTypeConstraint<"(" # !interleave(!foreach(p, preds, p.pred), " && ") # ")">;
class OrDType<list<DTypeConstraint> preds> :
    DTypeConstraint<"(" # !interleave(!foreach(p, preds, p.pred), " || ") # ")">;

def BoundDType : DTypeConstraint<"$_self">;
def UnboundDType : DTypeConstraint<"!$_self">;
def BoolDType : DTypeConstraint<"$_self->isBool()">;
def IndexDType : DTypeConstraint<"$_self->isIndex()">;
def IntDType : DTypeConstraint<"$_self->isInt()">;
def SIntDType : DTypeConstraint<"$_self->isSInt()">;
def UIntDType : DTypeConstraint<"$_self->isUInt()">;
def FloatDType : DTypeConstraint<"$_self->isFloat()">;
def UI8DType : DTypeConstraint<"*$_self == KGENDType::ui8">;
def SI8DType : DTypeConstraint<"*$_self == KGENDType::si8">;
def AddressDType : DTypeConstraint<"$_self->isAddress()">;
def ArithmeticDType : DTypeConstraint<"$_self->isArithmetic()">;

class BoundAnd<DTypeConstraint pred> : AndDType<[BoundDType, pred]>;
class UnboundOr<DTypeConstraint pred> : OrDType<[UnboundDType, pred]>;

/// A SIMD type whose dtype conforms to a constraint.
class SIMDOfDType<DTypeConstraint constraint, string desc> :
  ConfinedType<POP_SIMDType,
    [CPred<!subst("$_self", "llvm::cast<POP::SIMDType>($_self).getResolvedDType()",
                 constraint.pred)>], desc> {
  let cppClassName = "::M::KGEN::POP::SIMDType";
}

def BoolSIMD : SIMDOfDType<BoundAnd<BoolDType>, "!pop.simd<N, bool>">;
def UnboundSIMD : SIMDOfDType<UnboundDType, "!pop.simd<N, invalid>">;
def IndexSIMD : SIMDOfDType<BoundAnd<IndexDType>, "!pop.simd<N, index>">;
def IntegerSIMD : SIMDOfDType<BoundAnd<IntDType>, "!pop.simd<N, integral>">;
def FloatSIMD : SIMDOfDType<BoundAnd<FloatDType>, "!pop.simd<N, float>">;
def AddressSIMD : SIMDOfDType<BoundAnd<AddressDType>, "!pop.simd<N, address>">;

def UnboundOrIntegerSIMD : SIMDOfDType<UnboundOr<IntDType>,
    "whose element type is either unbound or an integral dtype">;
def UnboundOrSignedSIMD : SIMDOfDType<UnboundOr<OrDType<[SIntDType, FloatDType]>>,
    "whose element type is either unbound or of signed dtype">;
def UnboundOrFloatSIMD : SIMDOfDType<UnboundOr<FloatDType>,
    "whose element type is either unbound or a floating-point dtype">;
def UnboundOrIndexOrSignedSIMD : SIMDOfDType<UnboundOr<OrDType<
                                          [SIntDType, IndexDType, FloatDType]>>,
    "whose element type is either unbound, index, or of signed dtype">;
def UnboundOrBoolOrIndexOrIntegerSIMD : SIMDOfDType<
    UnboundOr<OrDType<[IndexDType, IntDType, BoolDType]>>,
    "whose element type is either unbound, or integer, or bool">;
def UnboundOrIndexOrIntegerSIMD : SIMDOfDType<
    UnboundOr<OrDType<[IndexDType, IntDType]>>,
    "whose element type is either unbound, index, or integral dtype">;
def UnboundOrBoolOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<[BoolDType, ArithmeticDType]>>,
    "whose element type is either integer or float dtype">;
def UnboundOrArithmeticSIMD : SIMDOfDType<UnboundOr<ArithmeticDType>,
    "whose element type is either integer or float dtype">;
def UnboundOrIndexOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<[IndexDType, ArithmeticDType]>>,
    "whose element type is either unbound, index, or arithmetic dtype">;
def UnboundOrBoolOrIndexOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<[IndexDType, IntDType, BoolDType, FloatDType]>>,
    "whose element type is either unbound, integral, bool or float dtype">;
def UnboundOrAddressOrBoolOrIndexOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<
      [IndexDType, AddressDType, IntDType, BoolDType, FloatDType]>>,
    "whose element type is either unbound, address, integral, bool or float "
      "dtype">;

/// A "scalar" type is a SIMD of size 1.
def ScalarType : ConfinedType<POP_SIMDType,
    [CPred<"llvm::cast<POP::SIMDType>($_self).getResolvedSize() && "
           "*llvm::cast<POP::SIMDType>($_self).getResolvedSize() == 1">],
    "SIMD with size 1">;

/// A scalar type whose dtype conforms to a constraint.
class ScalarOfDType<DTypeConstraint constraint, string desc> :
  AllOfType<[SIMDOfDType<constraint, desc>, ScalarType], desc>;

def BoolScalar : ScalarOfDType<BoundAnd<BoolDType>, "!pop.simd<1, bool>">,
    BuildableType<"POP::SIMDType::get(1, DTypeConstantAttr::get("
                                  "$_builder.getContext(), KGENDType::bool))">;
def UnboundOrArithmeticScalar : ScalarOfDType<UnboundOr<ArithmeticDType>,
    "whose type is an arithmetic dtype">;

def UI8Scalar : ScalarOfDType<BoundAnd<UI8DType>, "!pop.simd<1, ui8>">,
    BuildableType<"POP::SIMDType::get(1, DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGENDType::ui8))">;
def SI8Scalar : ScalarOfDType<BoundAnd<SI8DType>, "!pop.simd<1, si8>">,
    BuildableType<"POP::SIMDType::get(1, DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGENDType::si8))">;

def UnboundOrPointerOrSIMDAddressType : AnyTypeOf<
    [PointerType, AddressSIMD, UnboundSIMD]>;


def UnboundOrIndexOrInt : AnyTypeOf<[
    ScalarOfDType<UnboundOr<OrDType<[IndexDType, IntDType]>>,
                  "whose type is unbound, index, or integral dtype">]>;

//===----------------------------------------------------------------------===//
// Type Constraint Traits

// Matches if Value has size=1 and same dtype as Operand
class SIMDSizeOneOfSameDType<string operand, string value> : TypesMatchWith<
    value # " type matches dtype of the " # operand # " operand", operand,
    value, "POP::SIMDType::get(1, llvm::cast<SIMDType>($_self).getDType())">;

class SIMDSize<string name> :
    StrFunc<"$" # name # ".getType().cast<POP::SIMDType>().getResolvedSize()">;

class SIMDOfSameSize<string lhs, string rhs> :
  AllMatchSameOperatorTrait<[lhs, rhs], SIMDSize<"_self">.result, "size">;

class AddressSIMDOfSameSize<string operand, string value> : TypesMatchWith<
    value # " type matches simd type of the " # operand # " operand", operand,
    value, "POP::SIMDType::get(llvm::cast<SIMDType>($_self).getSize(), "
    "DTypeConstantAttr::get($_self.getContext(), KGENDType::address))">;

class BoolSIMDOfSameSize<string operand, string value> : TypesMatchWith<
    value # " type matches simd type of the " # operand # " operand", operand,
    value, "POP::SIMDType::get(llvm::cast<SIMDType>($_self).getSize(), "
    "DTypeConstantAttr::get($_self.getContext(), KGENDType::kBool))">;

#endif // KGEN_POPDIALECT_POPTYPES_TD
