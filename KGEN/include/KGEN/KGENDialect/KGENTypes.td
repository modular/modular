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

def MLIRTypeType : KGENType<"MLIRType"> {
  let summary = "Arbitrary MLIR type";
  let description = [{
    This kgen type represents an arbitrary MLIR type, used for parameterizing
    type-generic operations.  It cannot be materialized into an SSA value,
    because it has no runtime representation.
  }];

  let mnemonic = "type";
  let parameters = (ins);
}

def DTypeType : KGENType<"DType"> {
  let summary = "8-bit integer DType type";
  let description = [{
    This type corresponds to the DType runtime class, representing an
    element type (aka "dtype") specifier for a data storage types.
  }];

  let mnemonic = "dtype";
  let parameters = (ins);
}
