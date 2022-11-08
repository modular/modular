//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef HLCFOPS_TD
#define HLCFOPS_TD

include "Support/HLCFDialect/HLCFDialect.td"
include "mlir/IR/OpBase.td"

class HLCFOp<string mnemonic, list<Trait> traits = []> :
    Op<HLCFDialect, mnemonic, traits>;

/// Placeholder to make TableGen happy. Will be deleted when ops are added.
class Placeholder : HLCFOp<"placeholder">;

#endif // HLCFOPS_TD
