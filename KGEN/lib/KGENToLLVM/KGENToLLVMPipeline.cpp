//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/ForwardDecls.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

void M::KGEN::buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                                       const LowerToLLVMOptions &options) {
  // Run the canonicalizer before the lowering passes.
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(createLowerKGENToPOP());

  SmallVector<std::string, 1> topLevelKernels;
  if (options.topLevelKernel.hasValue())
    topLevelKernels.push_back(options.topLevelKernel);

  LowerKGENToLLVMOptions kgenToLLVMOptions{/*indexBitwidth=*/0,
                                           topLevelKernels};
  pm.addPass(createLowerKGENToLLVM(kgenToLLVMOptions));

  // Run all LLVM lowering passes.
  pm.addPass(createLowerGlobalPOPToLLVM());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(createLowerPOPToLLVM());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(createLowerSCFToLLVM());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
      mlir::createConvertIndexToLLVMPass());

  // And finally canonicalize again.
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createCanonicalizerPass());
}

void M::KGEN::registerLowerToLLVMPipeline() {
  mlir::PassPipelineRegistration<LowerToLLVMOptions>(
      "lower-to-llvm", "Lower KGEN IR to LLVM IR.", buildLowerToLLVMPipeline);
}
