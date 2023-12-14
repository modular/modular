//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

void KGEN::buildCheckLITPipeline(mlir::PassManager &pm, LLCL::Runtime &runtime,
                                 const CompilationOptions &options) {
  // Lower semantic control flow operations like lit.return to terminators and
  // diagnose unreachable code.
  pm.addPass(createLowerSemanticCF());
  pm.addPass(createVerifyParameters());

  // These passes doesn't touch parameters, no need to re-verify them after it.

  // Check if a struct contains recursive nested struct fields and emit error if
  // found.
  pm.addPass(createCheckRecursiveStructs());

  // Insert calls to destructors, reject use before free, and borrow check.
  pm.addPass(createCheckLifetimes());
}

void KGEN::buildGenerateLibraryPipeline(mlir::PassManager &pm,
                                        LLCL::Runtime &runtime,
                                        const CompilationOptions &options) {
  buildCheckLITPipeline(pm, runtime, options);

  pm.addPass(createLowerLIT(
      {static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage)}));
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerLITTypes());
  pm.addPass(createVerifyParameters());
  // Eliminate dead symbols. If we don't use the symbol *somewhere* it doesn't
  // need to be in the IR.
  pm.addPass(createEliminateDeadSymbols());

  // Only inline `always_inline_no_debug` functions during parametric inlining.
  // Too much inlining pre-elaboration increases pressure on the elaborator and
  // reduces cache granularity. By restricting inlining to `nodebug` functions,
  // we still maintain the zero-cost abstraction.
  InlineParametricOptions inlinerOpts;
  inlinerOpts.nodebugOnly = true;
  inlinerOpts.optimizationLevel = options.optimizationLevel;
  inlinerOpts.updateDebugInfo =
      options.debugLevel != CompilationOptions::DebugInfoLevel::kNoDebug;
  pm.addPass(createInlineParametric(runtime, inlinerOpts));
  if (options.optimizationLevel >= 1) {
    pm.addPass(createVerifyParameters(
        VerifyParametersOptions{/*simplifyParameters=*/true}));
  }

  // These passes don't influence parameters, so we don't need to verify them.

  // We use the canonicalizer, but disable region simplifications, since it is
  // very CFG centric and we have region trees with a single block per region.
  if (options.optimizationLevel >= 1) {
    pm.addNestedPass<GeneratorOp>(createSROA());
    pm.addNestedPass<GeneratorOp>(createMem2Reg());
    pm.addNestedPass<GeneratorOp>(createSCCP());
    pm.addNestedPass<GeneratorOp>(createCanonicalizer());
    pm.addNestedPass<GeneratorOp>(createConstraintReduction());
  }

  pm.addPass(MOGGPreElab::createSliceMOGGFuncs());
}

void KGEN::buildElaborateModulePipeline(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    BuildInfoAttr build, const CompilationOptions &options,
    EvaluatorExecutorFn evaluatorExecutorFn,
    ElaboratorCompileAsmFn compileAsmFn,
    PackageLinkHandlerFn packageLinkHandlerFn) {
  // At the end of the LIT lowering pipeline, pull in the bodies of constructs
  // that were already elaborated.
  pm.addPass(createMaterializePackages(packageLinkHandlerFn));

  // Erase debuginfo from all sources if compiling with no debuginfo.
  if (options.debugLevel == CompilationOptions::kNoDebug)
    pm.addPass(DebugInfo::createDebugInfoStrip());

  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures());
  // TODO(#20717): CSE cannot run before `OutlineClosures`.
  pm.addNestedPass<GeneratorOp>(mlir::createCSEPass());
  pm.addPass(createVerifyParameters());
  pm.addPass(createLiftAndFoldApply());

  // After elaboration, we have no use for the parameter verifier anymore.
  ElaborateGeneratorsOptions elaboratorOptions;
  elaboratorOptions.enableSearch = options.enableSearch;
  elaboratorOptions.elaborateDebugInfo =
      options.debugLevel == CompilationOptions::kLineTablesOnly ||
      options.debugLevel == CompilationOptions::kFullDebugInfo;
  elaboratorOptions.diagAllFailures = options.emitAllElaboratorDiags;
  pm.addPass(createElaborateGenerators(
      runtime, target, build, elaboratorOptions, std::move(evaluatorExecutorFn),
      std::move(compileAsmFn)));
}

void KGEN::buildPostElaborationPipeline(mlir::PassManager &pm,
                                        LLCL::Runtime &runtime,
                                        const CompilationOptions &options) {
  // Run DCE first coming out of the elaborator.
  pm.addPass(createEliminateDeadSymbols());

  // Then immediately resolve compiler promises.
  pm.addPass(createResolveCompilerPromises(runtime));

  // We lower argument input conventions.
  pm.addPass(createLowerInputConventions());
  // TODO(#20700): should this be followed by mem-2-reg immediately?

  // Run the ForceInline pass with an inner function pass pipeline.
  auto buildForceInlineFuncPasses = [options](mlir::OpPassManager &pm) {
    if (options.optimizationLevel < 1)
      return;
    pm.addPass(createSimplifyCF());
    pm.addPass(createSROA());
    pm.addPass(createMem2Reg());
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
      runtime,
      {options.debugLevel != CompilationOptions::DebugInfoLevel::kNoDebug},
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
      runtime,
      {options.debugLevel != CompilationOptions::DebugInfoLevel::kNoDebug,
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
  // At the end of the pipeline, externalize any functions that have been
  // precompiled so that they aren't sent to LLVM again.
  pm.addPass(createExternalizePrecompiledFunctions());
}
