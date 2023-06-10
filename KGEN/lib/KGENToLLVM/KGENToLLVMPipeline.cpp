//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/ForwardDecls.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

void M::KGEN::buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                                       const LowerToLLVMOptions &options) {
  using mlir::LLVM::LLVMFuncOp;

  // Run all LLVM lowering passes.
  pm.addPass(createLowerKGENToLLVM());
  pm.addPass(createLowerRuntimeClosures());
  pm.addNestedPass<LLVMFuncOp>(createLowerPOPToLLVM());
  pm.addNestedPass<LLVMFuncOp>(createTweakSpilledAllocas());
  pm.addPass(createLowerKGENCoroutinesAsync());
  pm.addPass(createLowerGlobalPOPToLLVM());
  pm.addNestedPass<LLVMFuncOp>(createLowerControlFlow());

  // And finally canonicalize again.
  mlir::GreedyRewriteConfig cannConfig;
  cannConfig.enableRegionSimplification = false;
  pm.addNestedPass<LLVMFuncOp>(mlir::createCanonicalizerPass(cannConfig));
  pm.addNestedPass<LLVMFuncOp>(mlir::createCSEPass());

  // If requested, generate debug info at the LLVM level.
  if (options.debugAtLevel.hasValue() &&
      options.debugAtLevel == CompilationOptions::kDebugAtLLVM) {
    pm.addPass(DebugInfo::createDebugInfoSnapshot(
        {options.debugInfoLevel, /*filename*/ ""}));
  }

  // Run the LLVM lowering for debug info last.
  pm.addPass(DebugInfo::createDebugInfoToLLVM());
}

void M::KGEN::registerLowerToLLVMPipeline() {
  mlir::PassPipelineRegistration<LowerToLLVMOptions>(
      "lower-to-llvm", "Lower KGEN IR to LLVM IR.", buildLowerToLLVMPipeline);
}
