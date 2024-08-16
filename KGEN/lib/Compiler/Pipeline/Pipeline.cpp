//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Pipeline.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

void KGEN::buildCheckLITPipeline(mlir::PassManager &pm,
                                 const CompilationOptions &options) {
  // Lower semantic control flow operations like lit.return to terminators and
  // diagnose unreachable code.
  pm.addPass(createLowerSemanticCF());
#ifndef MODULAR_PRODUCTION
  pm.addPass(createVerifyParameters());
#endif

  // These passes doesn't touch parameters, no need to re-verify them after it.
  // Insert calls to destructors, reject use before free, and borrow check.
  pm.addPass(createCheckLifetimes());
}

/// This populates the early pipeline passes of the KGEN compiler that lowers
/// the lit dialect. It also runs the mogg-annotate pass. `
static void buildLowerLITPipeline(mlir::PassManager &pm,
                                  const CompilationOptions &options) {
  buildCheckLITPipeline(pm, options);

  pm.addPass(MOGGPreElab::createMOGGAnnotate());

  pm.addPass(createLowerLIT(
      {static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage)}));
#ifndef MODULAR_PRODUCTION
  pm.addPass(createVerifyParameters());
#endif

  pm.addPass(createLowerLITTypes());
#ifndef MODULAR_PRODUCTION
  pm.addPass(createVerifyParameters());
#endif
}

void KGEN::buildGenerateLibraryPipeline(
    mlir::PassManager &pm, const CompilationOptions &options,
    const CompileCanonicalizationFnsFn &compileCanonFn,
    KGENCompiler::StartPipelineAt startAt) {
  if (startAt == KGENCompiler::StartPipelineAt::Beginning) {
    // If the compilation options aren't for full debug, strip the extra info
    // from the module.
    if (options.debugLevel != CompilationOptions::kFullDebugInfo) {
      pm.addPass(DebugInfo::createDebugInfoStrip(
          {/*preserveLineTables=*/options.debugLevel ==
           CompilationOptions::kLineTablesOnly}));
    }

    buildLowerLITPipeline(pm, options);
  }

  // Slice MOGG compute & shape functions out of base kernels.
  MOGGPreElab::MOGGPreElabPipelineOptions moggOpts;
  moggOpts.debugBuild =
      options.debugLevel != CompilationOptions::DebugInfoLevel::kNoDebug;
  pm.addPass(MOGGPreElab::createMOGGPreElabPipeline(moggOpts));

  if (options.optimizationLevel >= 2)
    pm.addPass(createRemoveUnusedParams());

  // Eliminate dead symbols. If we don't use the symbol *somewhere* it doesn't
  // need to be in the IR.
  pm.addPass(createEliminateDeadSymbols());

  if (options.optimizationLevel >= 1) {
    pm.addNestedPass<GeneratorOp>(createSROA());
    pm.addNestedPass<GeneratorOp>(createMem2Reg());
    pm.addNestedPass<GeneratorOp>(createCanonicalizer(compileCanonFn));
  }

  // Only inline `always_inline_no_debug` functions during parametric inlining.
  // Too much inlining pre-elaboration increases pressure on the elaborator and
  // reduces cache granularity. By restricting inlining to `nodebug` functions,
  // we still maintain the zero-cost abstraction.
  InlineParametricOptions inlinerOpts;
  inlinerOpts.nodebugOnly = true;
  inlinerOpts.optimizationLevel = options.optimizationLevel;
  inlinerOpts.updateDebugInfo =
      options.debugLevel != CompilationOptions::DebugInfoLevel::kNoDebug;
  // FIXME(#32286): The bodies of precompiled functions are not available to the
  // parametric inliner.
  pm.addPass(createInlineParametric(inlinerOpts));
  pm.addPass(createVerifyParameters({/*simplifyParameters=*/true}));

  // These passes don't influence parameters, so we don't need to verify them.
  if (options.optimizationLevel >= 1) {
    if (options.optimizationLevel >= 2)
      pm.addPass(createRemoveUnusedParams());
    pm.addPass(createEliminateDeadSymbols());
    pm.addNestedPass<GeneratorOp>(createSROA());
    pm.addNestedPass<GeneratorOp>(createMem2Reg());
    pm.addNestedPass<GeneratorOp>(createSCCP());
    pm.addNestedPass<GeneratorOp>(createCanonicalizer(compileCanonFn));
    if (options.optimizationLevel >= 2)
      pm.addPass(createRemoveUnusedParams());
    pm.addPass(createEliminateDeadSymbols());
    pm.addPass(createApplyInliner());
    pm.addPass(createInlineParametric(inlinerOpts));
    pm.addPass(createVerifyParameters({/*simplifyParameters=*/true}));
    pm.addNestedPass<GeneratorOp>(createSROA());
    pm.addNestedPass<GeneratorOp>(createMem2Reg());
    pm.addNestedPass<GeneratorOp>(createCanonicalizer(compileCanonFn));
  }
}

void KGEN::buildElaborateModulePipeline(mlir::PassManager &pm,
                                        TargetInfoAttr target,
                                        const CompilationOptions &options,
                                        ElaboratorCompileAsmFn compileAsmFn) {
  pm.addPass(createEliminateDeadSymbols());

  // Erase debuginfo from all sources if compiling with no debuginfo.
  if (options.debugLevel == CompilationOptions::kNoDebug)
    pm.addPass(DebugInfo::createDebugInfoStrip());

  // Only outline closures just before elaboration - they aren't really
  // necessary until elaboration happens.
  pm.addPass(createOutlineClosures(OutlineClosuresOptions{
      options.debugLevel != CompilationOptions::kNoDebug}));
  if (options.optimizationLevel >= 1) {
    // TODO(#20717): CSE cannot run before `OutlineClosures`.
    pm.addNestedPass<GeneratorOp>(mlir::createCSEPass());
  }
#ifndef MODULAR_PRODUCTION
  pm.addPass(createVerifyParameters());
#endif
  pm.addPass(createLiftAndFoldApply());

  // After elaboration, we have no use for the parameter verifier anymore.
  ElaborateGeneratorsOptions elaboratorOptions;
  elaboratorOptions.enableSearch = options.enableSearch;
  elaboratorOptions.elaborateDebugInfo =
      options.debugLevel == CompilationOptions::kLineTablesOnly ||
      options.debugLevel == CompilationOptions::kFullDebugInfo;
  elaboratorOptions.diagAllFailures = options.emitAllElaboratorDiags;
  pm.addPass(createElaborateGenerators(target, elaboratorOptions,
                                       std::move(compileAsmFn)));
}

void KGEN::buildPostElaborationPipeline(
    mlir::PassManager &pm, const CompilationOptions &options,
    const CompileCanonicalizationFnsFn &compileCanonFn) {
  // Run DCE first coming out of the elaborator.
  pm.addPass(createEliminateDeadSymbols());

  // Then immediately resolve compiler promises.
  pm.addPass(createResolveCompilerPromises());

  // We lower argument input conventions.
  pm.addNestedPass<FuncOp>(createLowerArgConventions());
  pm.addNestedPass<FuncOp>(createLowerCallingConventions());
  pm.addNestedPass<FuncOp>(createCanonicalizer(compileCanonFn));
  pm.addNestedPass<FuncOp>(createMem2Reg());

  // Run the AutomaticInline pass with an inner function pass pipeline.
  auto buildAutomaticInlineFuncPasses =
      [options, compileCanonFn](mlir::OpPassManager &pm) {
        if (options.optimizationLevel < 1)
          return;
        pm.addPass(createSimplifyCF());
        pm.addPass(createSROA());
        pm.addPass(createMem2Reg());
        // NOTE: Super important that ConditionPropagation pattern runs before
        // HoistTrivialInvariants, or else structural conditionals are lost.
        pm.addPass(createCanonicalizer(compileCanonFn));
        pm.addPass(mlir::createCSEPass());
        pm.addPass(createHoistTrivialInvariants());
        pm.addPass(createCanonicalizer(compileCanonFn));
        pm.addPass(createSimplifyCF());
        pm.addPass(createSROA());
        pm.addPass(createMem2Reg());
        pm.addPass(createStackReuse());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(createCanonicalizer(compileCanonFn));
      };

  pm.addPass(createAutomaticInline(
      {options.debugLevel == CompilationOptions::DebugInfoLevel::kNoDebug
           ? InlinerDebugInfoUpdateTime::kNever
           : (options.optimizationLevel == 0
                  ? InlinerDebugInfoUpdateTime::kDeferred
                  : InlinerDebugInfoUpdateTime::kImmediate),
       options.optimizationLevel},
      std::move(buildAutomaticInlineFuncPasses)));

  // Process debuginfo based on the selected debugging level.
  if (options.debugLevel == CompilationOptions::DebugInfoLevel::kSynthetic)
    pm.addPass(createSynthesizeDebugInfo(
        {static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage)}));

  // Guaranteed optimizations.
  pm.addNestedPass<FuncOp>(createSROA());
  pm.addNestedPass<FuncOp>(createMem2Reg());
  pm.addNestedPass<FuncOp>(createCanonicalizer(compileCanonFn));
  pm.addNestedPass<FuncOp>(createRaiseForLoops());
  pm.addNestedPass<FuncOp>(createLoopUnrolling({options.optimizationLevel}));

  if (options.optimizationLevel >= 1) {
    pm.addNestedPass<FuncOp>(createSROA());
    pm.addNestedPass<FuncOp>(createMem2Reg());
    pm.addNestedPass<FuncOp>(createSCCP());
    pm.addNestedPass<FuncOp>(createCanonicalizer(compileCanonFn));
    pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  }

  pm.addPass(createEliminateDeadSymbols());

  if (options.optimizationLevel >= 1) {
    pm.addPass(createArgPromotion());
    pm.addNestedPass<FuncOp>(createCanonicalizer(compileCanonFn));
    pm.addNestedPass<FuncOp>(createRaiseForLoops());
    pm.addNestedPass<FuncOp>(createLoopUnrolling({options.optimizationLevel}));
    pm.addNestedPass<FuncOp>(createSROA());
    pm.addNestedPass<FuncOp>(createMem2Reg());
    pm.addNestedPass<FuncOp>(createCanonicalizer(compileCanonFn));
    pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  }

  if (options.optimizationLevel >= 2) {
    pm.addNestedPass<FuncOp>(createRaiseForLoops());
    pm.addNestedPass<FuncOp>(createLoopUnrolling({options.optimizationLevel}));
    pm.addNestedPass<FuncOp>(createSROA());
    pm.addNestedPass<FuncOp>(createMem2Reg());
    pm.addNestedPass<FuncOp>(createSimplifyCF());
    pm.addNestedPass<FuncOp>(createCanonicalizer(compileCanonFn));
    pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  }

  pm.addNestedPass<FuncOp>(createLowerLoops());
  // Lower async functions and closures as late as possible.
  pm.addPass(createLowerClosures());
  pm.addPass(createLowerAsyncFunctions());
  // TODO (MOCO-805): re-enable dead-arg elimination
  // if (options.optimizationLevel >= 2)
  //   pm.addPass(createDeadArgumentElimination());
  pm.addPass(createEliminateDeadSymbols());
}
