//===- index-opt.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/IndexDialect/IndexDialect.h"
#include "Support/IndexToLLVM/IndexToLLVM.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, index::IndexDialect>();
  M::index::registerIndexToLLVMPass();
  mlir::registerCanonicalizer();
  return failed(
      mlir::MlirOptMain(argc, argv, "index optimizer driver", registry));
}
