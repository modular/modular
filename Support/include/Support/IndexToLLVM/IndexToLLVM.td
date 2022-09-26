//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def IndexToLLVM : Pass<"convert-index-to-llvm"> {
  let summary = "Lower the `index` dialect to the `llvm` dialect.";
  let description = [{
    This pass lowers Index dialect operations to LLVM dialect operations.
    Operation conversions are 1-to-1 except for the exotic divides: `ceildivs`,
    `ceildivu`, and `floordivs`, which expand to series of LLVM operations.
    Importantly, the index bitwidth should be correctly set to the target
    pointer width via `index-bitwidth`.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}
