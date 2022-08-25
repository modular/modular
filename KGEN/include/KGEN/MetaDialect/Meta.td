//===- Meta.td ------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Meta dialect.
//
//===----------------------------------------------------------------------===//

#ifndef META_TD
#define META_TD

include "mlir/IR/OpBase.td"
include "KGEN/MetaDialect/MetaDialect.td"

// Base class for the operation in this dialect.
class MetaOp<string mnemonic, list<Trait> traits = []> :
    Op<MetaDialect, mnemonic, traits>;

include "KGEN/MetaDialect/MetaTypes.td"
include "KGEN/MetaDialect/MetaOps.td"

#endif // KGEN_TD
