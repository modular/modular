//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry
      .insert<mlir::func::FuncDialect, mlir::index::IndexDialect,
              mlir::LLVM::LLVMDialect, DebugInfo::DebugInfoDialect, MDialect>();
  mlir::registerCanonicalizer();
  DebugInfo::registerDebugInfoToLLVMPass();
  DebugInfo::registerTransformsPasses();

  return failed(
      mlir::MlirOptMain(argc, argv, "index optimizer driver", registry));
}
