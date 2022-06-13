//===- KGEN.td ---------------------------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TD
#define KGEN_TD

include "mlir/IR/OpBase.td"
include "KGEN/KGENDialect/KGENDialect.td"

// Base class for the operation in this dialect.
class KGENOp<string mnemonic, list<Trait> traits = []> :
    Op<KGENDialect, mnemonic, traits>;

include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/KGENDialect/KGENOps.td"

#endif // KGEN_TD
