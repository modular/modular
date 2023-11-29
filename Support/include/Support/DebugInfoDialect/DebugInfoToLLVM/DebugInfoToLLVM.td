//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def DebugInfoToLLVM : Pass<"convert-debuginfo-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower the `debuginfo` dialect to the `llvm` dialect.";
  let description = [{
    This pass lowers Debug Info dialect attributes, operations, and types to
    the LLVM dialect.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}
