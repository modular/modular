//===- kgen-opt.cpp -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The kgen-opt driver implementation.
//
//===----------------------------------------------------------------------===//

#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENPasses.h"
#include "Support/IndexDialect/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<index::IndexDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect>();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  KGEN::registerPasses();

  return failed(mlir::MlirOptMain(argc, argv, "kgen optimizer driver", registry,
                                  /*preloadDialectsInContext=*/true));
}
