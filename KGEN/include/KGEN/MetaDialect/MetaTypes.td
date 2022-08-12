//===- MetaTypes.td -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the Meta dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class MetaType<string name, list<Trait> traits = []>
  : TypeDef<MetaDialect, name, traits> { }

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

def ScalarType : MetaType<"Scalar",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface,
                                   ["replaceImmediateSubAttribute"]>]> {
  let summary = "parameterized scalar type";
  let description = [{
    This type is parameterized with a value of !kgen.dtype type.
  }];

  let mnemonic = "scalar";
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

def SIMDType : MetaType<"SIMD",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface,
                                   ["replaceImmediateSubAttribute"]>]> {
  let summary = "parameterized SIMD vector type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.
  }];

  let mnemonic = "simd";
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
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

def BufferType : MetaType<"Buffer",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface,
                                   ["replaceImmediateSubAttribute"]>]> {
  let summary = "parameterized buffer type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.  The size
    field is optional (may be null) in which case this is a buffer with an
    unknown size.  The dtype type is also optional, in which case this is a
    buffer with unknown element type.
  }];

  let mnemonic = "buffer";
  let parameters = (ins "TypedAttr":$size, "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<OptionalIndexParamValue>($size) `,` ` `
        custom<OptionalParamDTypeValue>($dType) `>`
  }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def PointerType : MetaType<"Pointer",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface,
                                   ["replaceImmediateSubAttribute"]>]> {
  let summary = "parameterized pointer type";
  let description = [{
    This type is parameterized with a !kgen.dtype type. The dtype type is also
    optional, in which case this is a pointer to a buffer with unknown element
    type.
  }];

  let mnemonic = "pointer";
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
def BoolScalar : Type<
    CPred<"$_self.isa<ScalarType>() &&"
          "$_self.cast<ScalarType>().getDType().isa<DTypeConstantAttr>() &&"
          "$_self.cast<ScalarType>().getDType().cast<DTypeConstantAttr>().getDType().getValue() == DType::kBool">,
    "whose value is either unboard or a bool dtype">;

/// A simd type that is constrained to have a bool dtype.
def BoolSIMD : Type<
    CPred<"$_self.isa<SIMDType>() &&"
          "$_self.cast<SIMDType>().getDType().isa<DTypeConstantAttr>() &&"
          "$_self.cast<SIMDType>().getDType().cast<DTypeConstantAttr>().getDType().getValue() == DType::kBool">,
    "whose value is either unboard or a bool dtype">;

/// A scalar type that is constrained to be either an unbound dtype or a
/// floating-point dtype.
def UnboundOrFloatScalar : Type<
    CPred<"$_self.isa<ScalarType>() &&"
          "(!$_self.cast<ScalarType>().getDType().isa<DTypeConstantAttr>() ||"
          "$_self.cast<ScalarType>().getDType().cast<DTypeConstantAttr>().getDType().isFloat())">,
    "whose value is either unbound or a floating-point dtype">;

/// A simd type that is constrained to be either an unbound dtype or a
/// floating-point dtype.
def UnboundOrFloatSIMD : Type<
    CPred<"$_self.isa<SIMDType>() &&"
          "(!$_self.cast<SIMDType>().getDType().isa<DTypeConstantAttr>() ||"
          "$_self.cast<SIMDType>().getDType().cast<DTypeConstantAttr>().getDType().isFloat())">,
    "whose element type is either unbound or a floating-point dtype">;

def ScalarOrSIMD : AnyTypeOf<
    [ScalarType, SIMDType]>;

def BoolScalarOrSIMD : AnyTypeOf<
    [BoolScalar, BoolSIMD]>;

def UnboundOrFloatScalarOrSIMD : AnyTypeOf<
    [UnboundOrFloatScalar, UnboundOrFloatSIMD]>;
