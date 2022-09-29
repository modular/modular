//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MTYPES_TD
#define MTYPES_TD

include "Support/MDialect/M.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class MType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MDialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def M_ArrayType : MType<"Array", "array",
    [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "1D shaped type";
  let description = [{
    The `!M.array` type represents one dimensional data of known length. This
    type implements `ShapedType` and can be used with `ElementsAttr`.

    Example:

    ```mlir
    // An array of integers.
    !M.array<32xi32>

    // An array of floats.
    !M.array<256xf64>
    ```
  }];

  let parameters = (ins "int64_t":$size, "Type":$elementType);
  let assemblyFormat = "`<` custom<SizeX>($size) $elementType `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "int64_t":$size, "Type":$elementType)>
  ];
}

#endif // MTYPES_TD
