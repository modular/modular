//===- ConvertKGENToLLVM.td --------------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertKGENToLLVM : Pass<"convert-kgen-to-llvm", "ModuleOp"> {
  let summary = "Lower the `kgen` and `meta` dialects to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion from the fully-elaborated and specified
    kgen.kernel ops and kgen/meta types to the corresponding LLVM-dialect
    structures/types/ops.
  }];

  // We generate the LLVM dialect.
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let constructor = "::M::createConvertKGENToLLVMPass()";
}
