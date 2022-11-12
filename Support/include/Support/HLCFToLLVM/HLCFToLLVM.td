//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef HLCFTOLLVM_TD
#define HLCFTOLLVM_TD

include "mlir/Pass/PassBase.td"

def LowerHLCFToLLVMPass : Pass<"lower-hlcf-to-llvm"> {
  let summary = "Lower the `hlcf` dialect to `llvm`.";
  let description = [{
    The `lower-hlcf-to-llvm` pass lowers HLCF control-flow to CFGs. Regions
    and scope information is lost in this pass.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // HLCFTOLLVM_TD
