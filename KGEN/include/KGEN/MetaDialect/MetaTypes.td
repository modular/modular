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
      [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "parameterized scalar type";
  let description = [{
    This type is parameterized with a value of !kgen.dtype type.
  }];

  let mnemonic = "scalar";
  let parameters = (ins "::mlir::Attribute":$dtype);
  let assemblyFormat = "`<` custom<ParamDTypeValue>($dtype) `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

def SIMDType : MetaType<"SIMD",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "parameterized SIMD vector type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.
  }];

  let mnemonic = "simd";
  let parameters = (ins "::mlir::Attribute":$size, "::mlir::Attribute":$dtype);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` custom<ParamDTypeValue>($dtype) `>`
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

def BufferType : MetaType<"Buffer",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "parameterized buffer type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.  The size
    field is optional (may be null) in which case this is a buffer with an
    unknown size.
  }];

  let mnemonic = "buffer";
  let parameters = (ins "::mlir::Attribute":$size, "::mlir::Attribute":$dtype);
  let assemblyFormat = [{
    `<` custom<OptionalIndexParamValue>($size) `,` ` `
        custom<ParamDTypeValue>($dtype) `>`
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}
