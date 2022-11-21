//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DebugInfoSnapshot
//===----------------------------------------------------------------------===//

def DebugInfoSnapshot : Pass<"debuginfo-snapshot"> {
  let summary = "Snapshot debug information for the current IR.";
  let description = [{
    This pass allows for generating synthetic debug information from the IR
    during any stage of compilation, by snapshotting the IR to a file and using
    that file to generate information for the operations. This overwrites any
    existing debug information within the IR.

    An optional filename may be provided to use for the snapshot, otherwise
    an intermediate file is generated.
  }];

  let options = [
    Option<"filename", "filename", "std::string", /*default=*/"",
           "The filename to print the generated IR">
  ];
  let dependentDialects = ["DebugInfo::DebugInfoDialect"];
}
