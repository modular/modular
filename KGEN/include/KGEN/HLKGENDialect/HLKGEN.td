//===- HLKGEN.td -------------------------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the HLKGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HLKGEN_TD
#define HLKGEN_TD

include "mlir/IR/OpBase.td"
include "KGEN/HLKGENDialect/HLKGENDialect.td"

include "KGEN/KGENDialect/KGENDialect.td"
include "KGEN/KGENDialect/KGENAttrs.td"


// Base class for the operation in this dialect.
class HLKGENOp<string mnemonic, list<Trait> traits = []> :
    Op<HLKGENDialect, mnemonic, traits>;

include "KGEN/HLKGENDialect/HLKGENOps.td"

#endif // KGEN_TD
