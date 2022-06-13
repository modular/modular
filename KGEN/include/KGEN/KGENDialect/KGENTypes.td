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
