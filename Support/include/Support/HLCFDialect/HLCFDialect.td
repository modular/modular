//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef HLCFDIALECT_TD
#define HLCFDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// HLCFDialect
//===----------------------------------------------------------------------===//

def HLCFDialect : Dialect {
  let name = "HLCF";

  let summary = "high-level control-flow dialect";
  let description = [{
    The `HLCF` dialect implements a high-level control-flow paradigm that models
    control-flow across multiple operations and regions in the IR. The HLCF
    dialect aims to provide a structured and scoped representation of program
    control-flow with the same expressibility as standard control-flow graphs
    (CFGs). The HLCF dialect provides a model closer to source-level programs
    than pure structured control-flow and CFGs.
  }];

  let cppNamespace = "::M::HLCF";
}

#endif // HLCFDIALECT_TD
