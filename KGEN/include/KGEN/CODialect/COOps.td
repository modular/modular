//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef COOPS_TD
#define COOPS_TD

include "KGEN/CODialect/CODialect.td"
include "mlir/IR/OpBase.td"

class COOp<string mnemonic, list<Trait> traits = []> :
    Op<CODialect, mnemonic, traits>;

#endif // COOPS_TD
