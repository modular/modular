//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGENToLLVMPipeline.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
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
  pm.addNestedPass<LLVMFuncOp>(createLowerPOPToLLVM());
  // NOTE: Disable this pass because it is not correct for fine-grained lifetime
  // markers. It will be deleted soon anyways.
  // pm.addNestedPass<LLVMFuncOp>(createTweakSpilledAllocas());
  pm.addPass(createLowerKGENCoroutinesAsync());
  pm.addPass(createLowerGlobalPOPToLLVM(LowerGlobalPOPToLLVMOptions{
      options.alignedAllocFnName, options.alignedFreeFnName}));
  pm.addNestedPass<LLVMFuncOp>(createLowerControlFlow());

  // And finally canonicalize again.
  // FIXME(#25742): The MLIR region simplifier has exponential behaviour.
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  pm.addNestedPass<LLVMFuncOp>(mlir::createCanonicalizerPass(config));
  pm.addNestedPass<LLVMFuncOp>(mlir::createCSEPass());

  // If requested, generate debug info at the LLVM level.
  if (options.debugAtLevel.hasValue() &&
      options.debugAtLevel == CompilationOptions::kDebugAtLLVM) {
    pm.addPass(DebugInfo::createDebugInfoSnapshot(
        {options.debugInfoLevel, /*filename*/ "", options.debugInfoLanguage}));
  }

  // Run the LLVM lowering for debug info last.
  pm.addPass(DebugInfo::createDebugInfoToLLVM());
}

void KGEN::registerLowerToLLVMPipeline() {
  mlir::PassPipelineRegistration<LowerToLLVMOptions>(
      "lower-to-llvm", "Lower KGEN IR to LLVM IR.", buildLowerToLLVMPipeline);
}
