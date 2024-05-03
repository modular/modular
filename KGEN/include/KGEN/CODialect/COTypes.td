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
    underlying type is an opaque pointer. This type contains the return types
    of the coroutine.

    Example:

    ```mlir
    // A coroutine that returns two results: an index and a 32-bit integer.
    !co.routine<index, i32>
    // A coroutine that throws.
    !co.routine<i32>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"Type">:$types);
  let assemblyFormat = "`<` $types `>`";
  let assemblyFormat = "`<` (`>`) : ($types^ `>`)?";
}

class CoroutineResultTypes<string name, string ref> : TypesMatchWith<
    name # " types match coroutine result types", "coroutine", ref,
    "$_self.cast<CoroutineType>().getTypes()">;

def OpaqueCoroutineHandle : PointerOf<I8>;

#endif // COTYPES_TD
