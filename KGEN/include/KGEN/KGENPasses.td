//===- KGENPasses.td ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def LowerHLKGEN : Pass<"lower-hlkgen", "mlir::ModuleOp"> {
  let summary = "Lower the `hlkgen` dialects to the `kgen` dialect.";
  let description = [{
    This pass performs lowers high level abstractions from "hlkgen" to "kgen"
    and diagnoses any errors that are found along the way.
  }];

  let dependentDialects = ["KGENDialect"];

  let constructor = "::M::KGEN::createLowerHLKGENPass()";
}

def ConvertKGENToLLVM : Pass<"convert-kgen-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower the `kgen` and `meta` dialects to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion from the fully-elaborated and specified
    kgen.kernel ops and kgen/meta types to the corresponding LLVM-dialect
    structures/types/ops.
  }];

  // We generate the LLVM dialect.
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let constructor = "::M::KGEN::createConvertKGENToLLVMPass()";

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

def ConvertPOPToLLVM : Pass<"convert-pop-to-llvm", "KernelOp"> {
  let summary = "Lower the `pop` dialect to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion of fully-elaborated and specified `pop`
    dialect operations to LLVM dialect operations.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let constructor = "::M::KGEN::createConvertPOPToLLVMPass()";

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}
