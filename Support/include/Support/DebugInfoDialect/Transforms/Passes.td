//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DebugInfoStrip
//===----------------------------------------------------------------------===//

def DebugInfoStrip : Pass<"debuginfo-strip"> {
  let summary = "Strip debug information for the current IR.";
  let description = [{
    This pass strips all debuginfo dialect information from the current IR, but
    leaves normal location information in place (to still allow proper
    verification and error reporting).
  }];
  let options = [
    Option<"preserveLineTables", "preserveLineTables", "bool",
           /*default=*/"false",
           "Don't strip information used for line table generation.">,
  ];
}
