//===- MetaTypes.td -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the Meta dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/KGENDialect/KGENTypeInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/MetaDialect/MetaDialect.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class MetaType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MetaDialect, name, [
        DeclareTypeInterfaceMethods<OpaqueObjectInterface>
      ] # traits> {
  let mnemonic = typeMnemonic;
}

// Base class for parameterized types.
class ParameterizedMetaType<string name, string typeMnemonic,
                            list<Trait> traits = []>
    : MetaType<name, typeMnemonic, [
        DeclareTypeInterfaceMethods<SubElementTypeInterface>,
        DTypeInterface
      ] # traits>;

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

def ScalarType : ParameterizedMetaType<"Scalar", "scalar"> {
  let summary = "parameterized scalar type";
  let description = [{
    This type is parameterized with a value of !kgen.dtype type.
  }];

  let parameters = (ins "TypedAttr":$dType);
  let assemblyFormat = "`<` custom<ParamDTypeValue>($dType) `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$parameter), [{
      return $_get(parameter.getContext(), parameter);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

def SIMDType : ParameterizedMetaType<"SIMD", "simd"> {
  let summary = "parameterized SIMD vector type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.
  }];

  let parameters = (ins "TypedAttr":$size, "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` ` ` custom<ParamDTypeValue>($dType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size, "TypedAttr":$ty), [{
      return $_get(size.getContext(), size, ty);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the size expression of the SIMD type to an integer.
    Optional<int64_t> resolveSize() const;
  }];
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

def BufferType : ParameterizedMetaType<"Buffer", "buffer"> {
  let summary = "parameterized buffer type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.  The size
    field is optional (may be null) in which case this is a buffer with an
    unknown size.  The dtype type is also optional, in which case this is a
    buffer with unknown element type.
  }];

  let parameters = (ins "TypedAttr":$size, "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<OptionalIndexParamValue>($size) `,` ` `
        custom<OptionalParamDTypeValue>($dType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size, "TypedAttr":$ty), [{
      return $_get(size.getContext(), size, ty);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the size expression of the buffer type to an integer.
    Optional<int64_t> resolveSize() const;
  }];
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def PointerType : ParameterizedMetaType<"Pointer", "pointer"> {
  let summary = "parameterized pointer type";
  let description = [{
    This type is parameterized with a !kgen.dtype type. The dtype type is also
    optional, in which case this is a pointer to a buffer with unknown element
    type.
  }];

  let parameters = (ins "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<OptionalParamDTypeValue>($dType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Attribute":$ty), [{
      return $_get(ty.getContext(), ty);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

/// A scalar type that is constrained to have a bool dtype.
def BoolScalar : AllOfType<[ScalarType, BoolDType]>;
/// A simd type that is constrained to have a bool dtype.
def BoolSIMD : AllOfType<[SIMDType, BoolDType]>;

/// A scalar type that is constrained to be either an unbound dtype or an
/// integeral dtype.
def UnboundOrIntegerScalar : AllOfType<[ScalarType, IntegerOrUnboundDType],
    "whose value is either unbound or a integral dtype">;
/// A scalar type that is constrained to be either an unbound dtype or a
/// floating-point dtype.
def UnboundOrFloatScalar : AllOfType<[ScalarType, FloatOrUnboundDType],
    "whose value is either unbound or a floating-point dtype">;
/// A simd type that is constrained to be either an unbound dtype or an
/// integral dtype.
def UnboundOrIntegerSIMD : AllOfType<[SIMDType, IntegerOrUnboundDType],
    "whose element type is either unbound or a integral dtype">;
/// A simd type that is constrained to be either an unbound dtype or a
/// floating-point dtype.
def UnboundOrFloatSIMD : AllOfType<[SIMDType, FloatOrUnboundDType],
    "whose element type is either unbound or a floating-point dtype">;
/// A pointer to a concrete dtype.
def ConcretePointer : ConfinedType<PointerType,
    [CPred<"$_self.cast<PointerType>().getDType()">]>;
/// A buffer type with a concrete dtype.
def ConcreteBuffer : ConfinedType<BufferType,
    [CPred<"$_self.cast<BufferType>().getDType()">],
    "buffer with concrete dtype">;

def ScalarOrSIMD : AnyTypeOf<
    [ScalarType, SIMDType, ParamRefType]>;

def BoolScalarOrSIMD : AnyTypeOf<
    [BoolScalar, BoolSIMD, ParamRefType]>;

def UnboundOrFloatScalarOrSIMD : AnyTypeOf<
    [UnboundOrFloatScalar, UnboundOrFloatSIMD, ParamRefType]>;

def UnboundOrIntegerScalarOrSIMD : AnyTypeOf<
    [UnboundOrIntegerScalar, UnboundOrIntegerSIMD, ParamRefType]>;

/// Checks if the given value's type is a scalar type which matches the
/// element dtype of the oeprand.
class ScalarOfSameDType<string operand, string value> :
  TypesMatchWith< value # " type matches element type of the " # operand # " operand",
                  operand, value, "getScalarOfSameDType($_self)">;
