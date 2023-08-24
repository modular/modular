//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef POPTYPES_TD
#define POPTYPES_TD

include "KGEN/KGENDialect/KGENTypeInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/POPDialect/POPDialect.td"
include "Support/MDialect/MTypeInterfaces.td"
include "Support/Interpreter/InterpreterInterface.td"
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

  let parameters = (ins "TypedAttr":$size, "TypedAttr":$elementType);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` ` `
        custom<TypeParamValue>($elementType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size,
                                        "TypedAttr":$elementType)>,
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size,
                                        "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "int64_t":$size, "Type":$elementType)>,
    TypeBuilderWithInferredContext<(ins "ValueRange":$elements)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the size expression to an integer.
    std::optional<int64_t> getResolvedSize() const;
    /// Return the elementType attribute as a type.
    Type getElementAsType() const;
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class ArrayElementType<string lhs, string rhs> : TypesMatchWith<
    "value type matches array element type", lhs, rhs,
    "ParamRefType::get($_self.cast<POP::ArrayType>().getElementType())",
    "typeRangeMatches">;

def ArrayKnownSize : ConfinedType<POP_ArrayType,
    [CPred<"$_self.cast<POP::ArrayType>().getResolvedSize()">],
    "array with known size">;

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def POP_PointerType : POPType<"Pointer", "pointer",
    [DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "parameterized pointer type";
  let description = [{
    This type represents a pointer. The pointee type is parameterized with a
    `!kgen.mlirtype` type. An optional `addressSpace` argument can be specified
    (default to 0).

    Example:

    ```mlir
    // A pointer to a scalar.
    !pop.pointer<scalar<f32>>

    // A pointer to a SIMD vector.
    !pop.pointer<simd<4, f32>>

    // A parameterized scalar pointer.
    !pop.pointer<scalar<type>>

    // A completely parameterized pointer.
    !pop.pointer<elementType>

    // The address space parameter is optional, but can be specified.
    !pop.pointer<scalar<si32>, 2>

    // The address space also works on parametrized pointers.
    !pop.pointer<elementType, 5>
    ```
  }];

  let parameters = (ins "TypedAttr":$elementType,
                        DefaultValuedParameter<"TypedAttr",
                          "Builder($_ctxt).getIndexAttr(0)">:$addressSpace);
  let assemblyFormat = [{
    `<` custom<TypeParamValue>($elementType) ``
    (`,` custom<IndexParamValue>($addressSpace)^)? `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$elementType,
                                        CArg<"unsigned", "0">:$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        CArg<"unsigned", "0">:$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "TypedAttr":$elementType,
                                        "TypedAttr":$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$addressSpace)>,
  ];

  let extraClassDeclaration = [{
    /// Return the elementType attribute as a type.
    Type getElementAsType() const;

    /// Get the element type as the provided type.
    template <typename ElTy>
    ElTy getElementAs() { return getElementAsType().cast<ElTy>(); }
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class PointerElementType<string lhs, string rhs> : TypesMatchWith<
    "pointer element type", lhs, rhs,
    "ParamRefType::get($_self.cast<POP::PointerType>().getElementType())">;

class PointerOf<Type type> : ConfinedType<POP_PointerType,
    [CPred<"$_self.cast<POP::PointerType>().getElementType()">,
     SubstLeaves<"$_self",
      "$_self.cast<POP::PointerType>().getElementAsType()",
      type.predicate>],
    "pointer to " # type.summary> {
  let cppClassName = "::M::KGEN::POP::PointerType";
  let builderCall = !if(!empty(type.builderCall), "",
                       "PointerType::get(" # type.builderCall #")");
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
    [CPred<!subst("$_self", "$_self.cast<POP::SIMDType>().getResolvedDType()",
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
def UnboundOrArithmeticSIMD : SIMDOfDType<UnboundOr<ArithmeticDType>,
    "whose element type is either integer or float dtype">;
def UnboundOrIndexOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<[IndexDType, ArithmeticDType]>>,
    "whose element type is either unbound, index, or arithmetic dtype">;
def UnboundOrBoolOrIndexOrArithmeticSIMD : SIMDOfDType<
    UnboundOr<OrDType<[IndexDType, IntDType, BoolDType, FloatDType]>>,
    "whose element type is either unbound, integral, bool or float dtype">;

/// A "scalar" type is a SIMD of size 1.
def ScalarType : ConfinedType<POP_SIMDType,
    [CPred<"$_self.cast<POP::SIMDType>().getResolvedSize() && "
           "*$_self.cast<POP::SIMDType>().getResolvedSize() == 1">],
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
    [POP_PointerType, AddressSIMD, UnboundSIMD]>;


def UnboundOrIndexOrInt : AnyTypeOf<[
    ScalarOfDType<UnboundOr<OrDType<[IndexDType, IntDType]>>,
                  "whose type is unbound, index, or integral dtype">]>;

//===----------------------------------------------------------------------===//
// Type Constraint Traits

// Matches if Value has size=1 and same dtype as Operand
class SIMDSizeOneOfSameDType<string operand, string value> : TypesMatchWith<
    value # " type matches dtype of the " # operand # " operand", operand,
    value, "POP::SIMDType::get(1, $_self.cast<SIMDType>().getDType())">;

class SIMDSize<string name> :
    StrFunc<"$" # name # ".getType().cast<POP::SIMDType>().getResolvedSize()">;

class SIMDOfSameSize<string lhs, string rhs> :
  AllMatchSameOperatorTrait<[lhs, rhs], SIMDSize<"_self">.result, "size">;

class AddressSIMDOfSameSize<string operand, string value> : TypesMatchWith<
    value # " type matches simd type of the " # operand # " operand", operand,
    value, "POP::SIMDType::get($_self.cast<SIMDType>().getSize(), "
    "DTypeConstantAttr::get($_self.getContext(), KGENDType::address))">;

class BoolSIMDOfSameSize<string operand, string value> : TypesMatchWith<
    value # " type matches simd type of the " # operand # " operand", operand,
    value, "POP::SIMDType::get($_self.cast<SIMDType>().getSize(), "
    "DTypeConstantAttr::get($_self.getContext(), KGENDType::kBool))">;

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

def POP_StructType : POPType<"Struct", "struct", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized struct type";
  let description = [{
    This type represents a struct. A struct contains element types arranged in
    their order of declaration.

    Example:

    ```mlir
    // A struct with primitive element types.
    !pop.struct<scalar<f32>, simd<4, ui64>>

    // A struct with nested types.
    !pop.struct<
      !pop.pointer<simd<4, si8>>,
      !pop.array<24, scalar<si64>>,
      !pop.struct<
        !pop.scalar<f32>,
        !pop.scalar<f64>
      >
    >

    // A struct with parameterized element types.
    !pop.struct<type, array<size, scalar<dtype>>>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$elementTypes);
  let assemblyFormat = [{
    `<` (`>`) : (custom<TypeParamValues>($elementTypes)^ `>`)?
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilder<(ins "ArrayRef<Type>":$elementTypes)>,
    TypeBuilderWithInferredContext<(ins "ArrayRef<Type>":$elementTypes)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow all the element type expressions to MLIR types.
    LogicalResult
    resolveElementTypes(SmallVectorImpl<Type> &elementTypes) const;

    /// Get the number of struct elements.
    unsigned getNumElements() const { return getElementTypes().size(); }

    /// Get the element types as parameterized types.
    SmallVector<Type> getParameterizedElementTypes() const;

    /// Get a concrete element type.
    Type getConcreteElementType(unsigned i) const;
  }];
}

//===----------------------------------------------------------------------===//
// PackType
//===----------------------------------------------------------------------===//

def POP_PackType : POPType<"Pack", "pack", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized pack type";
  let description = [{
    A `!pop.pack` type represents a sequence of heterogeneously typed elements.
    This type can be used to represent a tuple of 0 or more elements.

    Example:

    ```mlir
    // A concrete pack type with no element types.
    !pop.pack<[]>

    // A concrete pack type with two element types.
    !pop.pack<[i32, i64]>

    kgen.generator @pack<Ts: variadic<!kgen.mlirtype>, T0: type, T1: type>(
      // A pack type parameterized on a variadic sequence of elements.
      %0: !pop,pack<Ts>,
      // A pack type parameterized on two element types.
      %1: !pop.pack<[T0, T1]>,
    ) { kgen.return }
    ```
  }];

  let parameters = (ins OptionalParameter<"TypedAttr">:$variadic);
  let assemblyFormat = [{
    `<` custom<ParamValue>(
      $variadic, "VariadicType::get(MLIRTypeType::get($_ctxt))") `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$variadic), [{
      return get(variadic.getContext(), variadic);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns whether this is a concrete pack type with no elements.
    bool isEmpty() const;

    /// If the pack type is concrete, returns the attribute representing the
    /// variadic sequence that the pack type is based upon. Otherwise, returns
    /// a null attribute.
    VariadicAttr getVariadicAttr() const;
  }];
}

//===----------------------------------------------------------------------===//
// VariantType
//===----------------------------------------------------------------------===//

def POP_VariantType : POPType<"Variant", "variant",
    [DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "A parametric variant type.";
  let description = [{
    A `!pop.variant` type is a structural variant type that represents a value
    that is one of a list of types (discriminated union).

    The type is always canonicalized to remove duplicate possible variant types.
    Variants of zero types or one type are allowed, e.g. `!pop.variant<i32>`,
    and all variant operations are defined on the even though they aren't
    particularly useful types.

    Example:

    ```mlir
    !pop.variant<i32, scalar<dtype>, T>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$types);
  let assemblyFormat = "`<` custom<TypeParamValues>($types) `>`";
  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilder<(ins "ArrayRef<TypedAttr>":$types)>,
    TypeBuilderWithInferredContext<(ins "ArrayRef<Type>":$types)>
  ];

  let extraClassDeclaration = [{
    /// Returns the index of the type in the variant type list. If the type is
    /// not in the list, returns None.
    std::optional<int64_t> getTypeIndex(Type type) const;

    /// Get the element types as parameterized types.
    SmallVector<Type> getParameterizedElementTypes() const;

    /// Get an element type as a parameterized type.
    Type getType(unsigned index);

    /// Return the number of types in the variant.
    size_t getNumTypes();
  }];
}

//===----------------------------------------------------------------------===//
// ClosureType
//===----------------------------------------------------------------------===//

def POP_ClosureType : POPType<"Closure", "closure"> {
  let summary = "The type for a closure.";
  let description = [{
    A type for a closure.

    Example:

    ```mlir
    !pop.closure<(index, i32) -> ()>
    ```
  }];

  let parameters = (ins "FunctionType":$func);
  let assemblyFormat = "`<` $func `>`";
}

//===----------------------------------------------------------------------===//
// CoroutineType
//===----------------------------------------------------------------------===//

def POP_CoroutineType : POPType<"Coroutine", "coroutine"> {
  let summary = "A coroutine handle.";
  let description = [{
    The `!pop.coroutine` type represents the handle to a coroutine. The
    underlying type is an opaque pointer. This type contains the return types
    of the coroutine.

    Example:

    ```mlir
    // A coroutine that returns two results: an index and a 32-bit integer.
    !pop.coroutine<() -> (index, i32)>
    // A coroutine that throws.
    !pop.coroutine<<>() throws -> i32>
    ```
  }];

  let parameters = (ins KGEN_SignatureType:$signature);
  let assemblyFormat = "`<` custom<KGENType>($signature) `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "SignatureType":$sig)>,
    TypeBuilderWithInferredContext<(ins "Type":$type), [{
      return get(
        type.getContext(), SignatureType::get(type.getContext(), {}, type));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get the result types of the coroutine.
    ArrayRef<Type> getResultTypes() { return getSignature().getValueResults(); }
  }];
}

class CoroutineResultTypes<string name, string ref> : TypesMatchWith<
    name # " types match coroutine result types", "coroutine", ref,
    "$_self.cast<CoroutineType>().getResultTypes()">;

def OpaqueCoroutineHandle : PointerOf<I8>;

#endif // POPTYPES_TD
