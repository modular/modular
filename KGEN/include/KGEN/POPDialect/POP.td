//===- POP.td ----------------------------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the POP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef POP_TD
#define POP_TD

include "mlir/IR/OpBase.td"
include "KGEN/POPDialect/POPAttrs.td"
include "KGEN/POPDialect/POPDialect.td"

// Base class for the operation in this dialect.
class POPOp<string mnemonic, list<Trait> traits = []> :
    Op<POPDialect, mnemonic, traits>;

include "KGEN/POPDialect/POPOps.td"

#endif // POP_TD
