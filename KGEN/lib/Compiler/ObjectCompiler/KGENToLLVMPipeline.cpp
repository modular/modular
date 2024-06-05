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

void M::KGEN::buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                                       const LowerToLLVMOptions &options) {
  using mlir::LLVM::LLVMFuncOp;

  // Run all LLVM lowering passes.
  pm.addPass(createLowerKGENToLLVM(LowerKGENToLLVMOptions{
      options.globalCtorFnName, options.globalDtorFnName}));
  pm.addPass(createLowerRuntimeClosures());
  pm.addNestedPass<LLVMFuncOp>(createLowerPOPToLLVM());
  pm.addNestedPass<LLVMFuncOp>(createTweakSpilledAllocas());
  pm.addPass(createLowerKGENCoroutinesAsync());
  pm.addPass(createLowerGlobalPOPToLLVM(LowerGlobalPOPToLLVMOptions{
      options.alignedAllocFnName, options.alignedFreeFnName}));
  pm.addNestedPass<LLVMFuncOp>(createLowerControlFlow());

  // And finally canonicalize again.
  // FIXME(#25742): The MLIR region simplifier has exponential behaviour.
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = false;
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

void M::KGEN::registerLowerToLLVMPipeline() {
  mlir::PassPipelineRegistration<LowerToLLVMOptions>(
      "lower-to-llvm", "Lower KGEN IR to LLVM IR.", buildLowerToLLVMPipeline);
}

// This is layer violation to have PostElaboration pipeline which is KGEN
// specific to be here in KGENToLLVM because lowerAllFuncsToLLVM calls this
// function for search. Consider cleaning this up when improving search.
void M::KGEN::buildPostElaborationPipeline(mlir::PassManager &pm,
                                           const CompilationOptions &options) {
  // Run DCE first coming out of the elaborator.
  pm.addPass(createEliminateDeadSymbols());

  // Then immediately resolve compiler promises.
  pm.addPass(createResolveCompilerPromises());

  // We lower argument input conventions.
  pm.addNestedPass<FuncOp>(createLowerArgConventions());
  pm.addNestedPass<FuncOp>(createLowerCallingConventions());
  pm.addNestedPass<FuncOp>(createMem2Reg());

  // Run the ForceInline pass with an inner function pass pipeline.
  auto buildForceInlineFuncPasses = [options](mlir::OpPassManager &pm) {
    if (options.optimizationLevel < 1)
      return;
    pm.addPass(createSimplifyCF());
    pm.addPass(createSROA());
    pm.addPass(createMem2Reg());
    // NOTE: Super important that ConditionPropagation pattern runs before
    // HoistTrivialInvariants, or else structural conditionals are lost.
    pm.addPass(createCanonicalizer());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createHoistTrivialInvariants());
    pm.addPass(createCanonicalizer());
    pm.addPass(createSimplifyCF());
    pm.addPass(createSROA());
    pm.addPass(createMem2Reg());
    pm.addPass(createStackReuse());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createCanonicalizer());
  };

  pm.addPass(createForceInline(
      {options.debugLevel == CompilationOptions::DebugInfoLevel::kNoDebug
           ? InlinerDebugInfoUpdateTime::kNever
           : (options.optimizationLevel == 0
                  ? InlinerDebugInfoUpdateTime::kDeferred
                  : InlinerDebugInfoUpdateTime::kImmediate),
       options.optimizationLevel},
      std::move(buildForceInlineFuncPasses)));

  // Process debuginfo based on the selected debugging level.
  if (options.debugLevel == CompilationOptions::DebugInfoLevel::kSynthetic)
    pm.addPass(createSynthesizeDebugInfo(
        {static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage)}));

  // Guaranteed optimizations.
  pm.addNestedPass<FuncOp>(createSROA());
  pm.addNestedPass<FuncOp>(createMem2Reg());
  pm.addNestedPass<FuncOp>(createCanonicalizer());
  pm.addNestedPass<FuncOp>(createRaiseForLoops());
  pm.addNestedPass<FuncOp>(createLoopUnrolling({options.optimizationLevel}));

  if (options.optimizationLevel >= 1) {
    pm.addNestedPass<FuncOp>(createSROA());
    pm.addNestedPass<FuncOp>(createMem2Reg());
    pm.addNestedPass<FuncOp>(createSCCP());
    pm.addNestedPass<FuncOp>(createCanonicalizer());
    pm.addNestedPass<FuncOp>(mlir::createCSEPass());
    pm.addNestedPass<FuncOp>(createFoldGlobalConstLoads());
  }

  pm.addPass(createEliminateDeadSymbols());

  // Run the AutomaticInliner with an inner function pass pipeline.
  auto buildAutomaticInlinerFuncPasses = [options](mlir::OpPassManager &pm) {
    if (options.optimizationLevel < 1)
      return;
    pm.addPass(createSimplifyCF());
    pm.addPass(createSROA());
    pm.addPass(createMem2Reg());
    // TODO: hoistTrivialInvariant is causing perf drop, needs further
    // investigation.
    // pm.addPass(createHoistTrivialInvariants());
    pm.addPass(createStackReuse());
    pm.addPass(createCanonicalizer());
    pm.addPass(mlir::createCSEPass());
  };

  pm.addPass(createAutomaticInline(
      {options.debugLevel == CompilationOptions::DebugInfoLevel::kNoDebug
           ? InlinerDebugInfoUpdateTime::kNever
           : (options.optimizationLevel == 0
                  ? InlinerDebugInfoUpdateTime::kDeferred
                  : InlinerDebugInfoUpdateTime::kImmediate),
       options.optimizationLevel},
      std::move(buildAutomaticInlinerFuncPasses)));

  if (options.optimizationLevel >= 1) {
    pm.addNestedPass<FuncOp>(createRaiseForLoops());
    pm.addNestedPass<FuncOp>(createLoopUnrolling({options.optimizationLevel}));
    pm.addNestedPass<FuncOp>(createSROA());
    pm.addNestedPass<FuncOp>(createMem2Reg());
    pm.addNestedPass<FuncOp>(createFoldGlobalConstLoads());
    pm.addNestedPass<FuncOp>(createCanonicalizer());
    pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  }

  if (options.optimizationLevel >= 2) {
    pm.addNestedPass<FuncOp>(createRaiseForLoops());
    pm.addNestedPass<FuncOp>(createLoopUnrolling({options.optimizationLevel}));
    pm.addNestedPass<FuncOp>(createSROA());
    pm.addNestedPass<FuncOp>(createMem2Reg());
    pm.addNestedPass<FuncOp>(createFoldGlobalConstLoads());
    pm.addNestedPass<FuncOp>(createSimplifyCF());
    pm.addNestedPass<FuncOp>(createCanonicalizer());
    pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  }

  pm.addNestedPass<FuncOp>(createLowerLoops());
  // Lower async functions and closures as late as possible.
  pm.addPass(createLowerClosures());
  if (options.optimizationLevel >= 2)
    pm.addPass(createDeadArgumentElimination());
  pm.addPass(createEliminateDeadSymbols());
}
