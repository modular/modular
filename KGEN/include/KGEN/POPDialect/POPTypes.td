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
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

class POPType<string typeName, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<POPDialect, typeName,
            [DeclareTypeInterfaceMethods<SubElementTypeInterface>] # traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def POP_ArrayType : POPType<"Array", "array"> {
  let summary = "parameterized array type";
  let description = [{
    This type represents a statically-sized array of elements. The size and
    element type are parameterized.

    Example:

    ```mlir
    // An array of four scalar floats.
    !pop.array<4, !pop.scalar<f32>>

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
    /// Try to narrow the size expression of the SIMD type to an integer.
    Optional<int64_t> getResolvedSize() const;
    /// Try to narrow the type expression of the pointer type to an MLIR type.
    Type getResolvedElementType() const;
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

def POP_PointerType : POPType<"Pointer", "pointer"> {
  let summary = "parameterized pointer type";
  let description = [{
    This type represents a pointer. The pointee type is parameterized with a
    `!kgen.mlirtype` type.

    Example:

    ```mlir
    // A pointer to a scalar.
    !pop.pointer<!pop.scalar<f32>>

    // A pointer to a SIMD vector.
    !pop.pointer<!pop.simd<4, f32>>

    // A parameterized scalar pointer.
    !pop.pointer<!pop.scalar<type>>

    // A completely parameterized pointer.
    !pop.pointer<elementType>
    ```
  }];

  let parameters = (ins "TypedAttr":$elementType);
  let assemblyFormat = [{
    `<` custom<TypeParamValue>($elementType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$elementType)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType)>,
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the type expression of the pointer type to an MLIR type.
    Type getResolvedElementType() const;
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class PointerElementType<string lhs, string rhs> : TypesMatchWith<
    "pointer element type", lhs, rhs,
    "ParamRefType::get($_self.cast<POP::PointerType>().getElementType())">;

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

def POP_ScalarType : POPType<"Scalar", "scalar", [DTypeInterface,
    DeclareTypeInterfaceMethods<OpaqueObjectInterface>]> {
  let summary = "parameterized scalar type";
  let description = [{
    This type represents a parameterized scalar value. The dtype is
    parameterized with a `!kgen.dtype`.

    Example:

    ```mlir
    // A floating point scalar.
    !pop.scalar<f32>

    // An unsigned integer scalar.
    !pop.scalar<ui64>

    // A parameterized scalar.
    !pop.scalar<dtype>
    ```
  }];

  let parameters = (ins "TypedAttr":$dType);
  let assemblyFormat = "`<` custom<DTypeParamValue>($dType) `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$dtype)>,
    TypeBuilder<(ins "DType":$dtype)>
  ];
}

//===----------------------------------------------------------------------===//
// TypeConstraints

def BoolScalar : AllOfType<[POP_ScalarType, BoolDType], "!pop.scalar<bool>">;

def UnboundOrIntegerScalar : AllOfType<[POP_ScalarType, IntegerOrUnboundDType],
    "whose value is either unbound or a integral dtype">;
def UnboundOrFloatScalar : AllOfType<[POP_ScalarType, FloatOrUnboundDType],
    "whose value is either unbound or a floating-point dtype">;

class ScalarOfSameDType<string operand, string value> : TypesMatchWith<
    value # " type matches dtype of the " # operand # " operand", operand,
    value, "POP::ScalarType::get($_self.cast<DTypeInterface>().getDType())">;

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

def POP_SIMDType : POPType<"SIMD", "simd", [DTypeInterface,
    DeclareTypeInterfaceMethods<OpaqueObjectInterface>]> {
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
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size, "TypedAttr":$dtype)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the size expression of the SIMD type to an integer.
    Optional<int64_t> getResolvedSize() const;
  }];
}

//===----------------------------------------------------------------------===//
// TypeConstraints

def BoolSIMD : AllOfType<[POP_SIMDType, BoolDType], "!pop.simd<N, bool>">;

def UnboundOrIntegerSIMD : AllOfType<[POP_SIMDType, IntegerOrUnboundDType],
    "whose element type is either unbound or a integral dtype">;
def UnboundOrFloatSIMD : AllOfType<[POP_SIMDType, FloatOrUnboundDType],
    "whose element type is either unbound or a floating-point dtype">;

def ScalarOrSIMD : AnyTypeOf<
    [POP_ScalarType, POP_SIMDType, ParamRefType]>;

def BoolScalarOrSIMD : AnyTypeOf<
    [BoolScalar, BoolSIMD, ParamRefType]>;

def UnboundOrFloatScalarOrSIMD : AnyTypeOf<
    [UnboundOrFloatScalar, UnboundOrFloatSIMD, ParamRefType]>;

def UnboundOrIntegerScalarOrSIMD : AnyTypeOf<
    [UnboundOrIntegerScalar, UnboundOrIntegerSIMD, ParamRefType]>;

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

def POP_StructType : POPType<"Struct", "struct"> {
  let summary = "parameterized struct type";
  let description = [{
    This type represents a struct. A struct contains element types arranged in
    their order of declaration.

    Example:

    ```mlir
    // A struct with primitive element types.
    !pop.struct<!pop.scalar<f32>, !pop.simd<4, ui64>>

    // A struct with nested types.
    !pop.struct<
      !pop.pointer<!pop.simd<4, si8>>,
      !pop.array<24, !pop.scalar<si64>>,
      !pop.struct<
        !pop.scalar<f32>,
        !pop.scalar<f64>
      >
    >

    // A struct with parameterized element types.
    !pop.struct<type, !pop.array<size, !pop.scalar<dtype>>>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$elementTypes);
  let assemblyFormat = "`<` custom<ArrayOfTypeExprs>($elementTypes) `>`";
  let genVerifyDecl = 1;

  let builders = [
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
  }];
}

//===----------------------------------------------------------------------===//
// VariantType
//===----------------------------------------------------------------------===//

def POP_VariantType : POPType<"Variant", "variant",
    [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
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
    !pop.variant<i32, !pop.scalar<dtype>, T>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$types);
  let assemblyFormat = "`<` custom<ArrayOfTypeExprs>($types) `>`";
  let skipDefaultBuilders = 1;

  let builders = [TypeBuilder<(ins "ArrayRef<TypedAttr>":$types)>];

  let extraClassDeclaration = [{
    /// Returns the index of the type in the variant type list. If the type is
    /// not in the list, returns None.
    Optional<int64_t> getTypeIndex(Type type) const;

    /// Get the element types as parameterized types.
    SmallVector<Type> getParameterizedElementTypes() const;
  }];
}

#endif // POPTYPES_TD
