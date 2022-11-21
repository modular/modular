//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/SnapshotDebugInfo.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/HLCFToLLVM/HLCFToLLVM.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  DebugInfo::DebugInfoDialect, HLCF::HLCFDialect, MDialect>();
  mlir::registerCanonicalizer();
  M::HLCF::registerLowerHLCFToLLVMPass();
  DebugInfo::registerDebugInfoToLLVMPass();
  DebugInfo::registerTransformsPasses();
  return failed(
      mlir::MlirOptMain(argc, argv, "index optimizer driver", registry));
}
