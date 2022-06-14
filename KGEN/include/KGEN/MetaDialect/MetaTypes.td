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

// Base class for other typedefs. Provides dialact-specific defaults.
class MetaType<string name> : TypeDef<MetaDialect, name> { }

// A parameterized scalar type.
def ScalarType : MetaType<"Scalar"> {
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
