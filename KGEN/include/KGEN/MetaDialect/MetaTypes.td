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
      ] # traits>;

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

def ScalarType : ParameterizedMetaType<"Scalar", "scalar", [DTypeInterface]> {
  let summary = "parameterized scalar type";
  let description = [{
    This type is parameterized with a value of !kgen.dtype type.
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
// SIMDType
//===----------------------------------------------------------------------===//

def SIMDType : ParameterizedMetaType<"SIMD", "simd", [DTypeInterface]> {
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

def ScalarOrSIMD : AnyTypeOf<
    [ScalarType, SIMDType, ParamRefType]>;

def BoolScalarOrSIMD : AnyTypeOf<
    [BoolScalar, BoolSIMD, ParamRefType]>;

def UnboundOrFloatScalarOrSIMD : AnyTypeOf<
    [UnboundOrFloatScalar, UnboundOrFloatSIMD, ParamRefType]>;

def UnboundOrIntegerScalarOrSIMD : AnyTypeOf<
    [UnboundOrIntegerScalar, UnboundOrIntegerSIMD, ParamRefType]>;

/// Checks if the given value's type is a scalar type which matches the element
/// dtype of the operand.
class ScalarOfSameDType<string operand, string value> :
  TypesMatchWith<value # " type matches dtype of the " # operand # " operand",
                 operand, value, "getScalarOfSameDType($_self)">;
