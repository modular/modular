//===- KGENTypes.td -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the KGEN dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults.
class KGENType<string name> : TypeDef<KGENDialect, name> { }

def DTypeType : KGENType<"DType"> {
  let summary = "8-bit integer TensorEltType type";
  let description = [{
    This type corresponds to the TensorEltType runtime class, representing an
    element type (aka "dtype") specifier for a data storage types.
  }];

  let mnemonic = "dtype";
  let parameters = (ins);
}

// A parameterized scalar type.
def ScalarType : KGENType<"Scalar"> {
  let summary = "parameterized scalar type";
  let description = [{
    Type type can be parameterized with an arbitrary !kgen.dtype type.
  }];

  let mnemonic = "scalar";
  let parameters = (ins "::mlir::Attribute":$dtype);
  let assemblyFormat = "`<` custom<ParamDTypeValue>($dtype) `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}
