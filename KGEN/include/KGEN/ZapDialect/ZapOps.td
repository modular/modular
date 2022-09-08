//===- ZapOps.td ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ZAPOPS_TD
#define ZAPOPS_TD

include "KGEN/ZapDialect/ZapDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ZapOp
//===----------------------------------------------------------------------===//

/// Base class for Zap dialect operations.
class ZapOp<string mnemonic, list<Trait> traits = []>
    : Op<ZapDialect, mnemonic, traits>;

// TODO: Remove this when ops are added. This placeholder prevents TableGen from
// complaining.
def PlaceholderOp : ZapOp<"placeholder">;

#endif // ZAPOPS_TD
