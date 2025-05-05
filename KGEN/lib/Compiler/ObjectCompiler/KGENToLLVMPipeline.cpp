//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGENToLLVMPipeline.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

void KGEN::buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                                    const LowerToLLVMOptions &options) {
  using mlir::LLVM::LLVMFuncOp;

  // Run all LLVM lowering passes.
  pm.addPass(createLowerKGENToLLVM(LowerKGENToLLVMOptions{
      options.globalCtorFnName, options.globalDtorFnName}));
  pm.addPass(createLowerRuntimeClosures());
  pm.addPass(createLowerGlobalPOPToLLVM());
  pm.addNestedPass<LLVMFuncOp>(createLegalizePOPOperations());
  pm.addNestedPass<LLVMFuncOp>(createLowerPOPToLLVM());
  pm.addNestedPass<LLVMFuncOp>(createLowerControlFlow());
  pm.addNestedPass<LLVMFuncOp>(mlir::createReconcileUnrealizedCastsPass());
  pm.addNestedPass<LLVMFuncOp>(createLowerSuspensionPoints());

  // And finally canonicalize again.
  // FIXME(#25742): The MLIR region simplifier has exponential behaviour.
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  pm.addNestedPass<LLVMFuncOp>(mlir::createCanonicalizerPass(config));
  pm.addNestedPass<LLVMFuncOp>(mlir::createCSEPass());

  // Run the LLVM lowering for debug info last.
  pm.addPass(DebugInfo::createDebugInfoToLLVM(
      {/*tradeoffPerfForVariableDI=*/options.optimizationLevel == 0}));
}

void KGEN::registerLowerToLLVMPipeline() {
  mlir::PassPipelineRegistration<LowerToLLVMOptions>(
      "lower-to-llvm", "Lower KGEN IR to LLVM IR.", buildLowerToLLVMPipeline);
}
