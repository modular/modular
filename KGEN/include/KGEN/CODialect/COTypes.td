//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef COTYPES_TD
#define COTYPES_TD

include "KGEN/CODialect/CODialect.td"
include "mlir/IR/AttrTypeBase.td"

include "KGEN/KGENDialect/KGENTypes.td"

class COType<string typeName, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<CODialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// CoroutineType
//===----------------------------------------------------------------------===//

def CoroutineType : COType<"Coroutine", "routine",
    [DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "A coroutine handle.";
  let description = [{
    The `!co.routine` type represents the handle to a coroutine. The
    underlying type is an opaque pointer.

    Example:

    ```mlir
    !co.routine
    ```
  }];

  let assemblyFormat = "";
}

#endif // COTYPES_TD
