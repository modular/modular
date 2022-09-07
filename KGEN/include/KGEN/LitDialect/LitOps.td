//===- LitOps.td ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LITOPS_TD
#define LITOPS_TD

include "KGEN/LitDialect/LitDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LitOp
//===----------------------------------------------------------------------===//

/// Base class for Lit dialect operations.
class LitOp<string mnemonic, list<Trait> traits = []>
    : Op<LitDialect, mnemonic, traits>;

// TODO: Remove this when ops are added. This placeholder prevents TableGen from
// complaining.
def PlaceholderOp : LitOp<"placeholder">;

#endif // LITOPS_TD
