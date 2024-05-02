//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef COTYPES_TD
#define COTYPES_TD

include "KGEN/CODialect/CODialect.td"
include "mlir/IR/AttrTypeBase.td"

class COType<string typeName, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<CODialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

#endif // COTYPES_TD
