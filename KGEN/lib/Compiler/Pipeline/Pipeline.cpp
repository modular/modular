//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Pipeline.h"
#include "KGEN/Compiler/KGENCompiler.h"
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
  pm.addPass(createVerifyParameters());

  // These passes doesn't touch parameters, no need to re-verify them after it.
  // Insert calls to destructors, reject use before free, and borrow check.
  pm.addPass(createCheckLifetimes());
}

void KGEN::buildGenerateLibraryPipeline(mlir::PassManager &pm,
                                        const CompilationOptions &options) {
  // If the compilation options aren't for full debug, strip the extra info from
  // the module.
  if (options.debugLevel != CompilationOptions::kFullDebugInfo) {
    pm.addPass(DebugInfo::createDebugInfoStrip(
        {/*preserveLineTables=*/options.debugLevel ==
         CompilationOptions::kLineTablesOnly}));
  }
  buildCheckLITPipeline(pm, options);

  pm.addPass(MOGGPreElab::createMOGGAnnotate());

  pm.addPass(createLowerLIT(
      {static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage)}));
  pm.addPass(createVerifyParameters());

  pm.addPass(createLowerLITTypes());
  pm.addPass(createVerifyParameters());

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
    pm.addNestedPass<GeneratorOp>(createCanonicalizer());
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
  pm.addPass(createVerifyParameters(
      VerifyParametersOptions{/*simplifyParameters=*/true}));

  // These passes don't influence parameters, so we don't need to verify them.
  if (options.optimizationLevel >= 1) {
    if (options.optimizationLevel >= 2)
      pm.addPass(createRemoveUnusedParams());
    pm.addPass(createEliminateDeadSymbols());
    pm.addNestedPass<GeneratorOp>(createSROA());
    pm.addNestedPass<GeneratorOp>(createMem2Reg());
    pm.addNestedPass<GeneratorOp>(createSCCP());
    pm.addNestedPass<GeneratorOp>(createCanonicalizer());
    if (options.optimizationLevel >= 2)
      pm.addPass(createRemoveUnusedParams());
    pm.addPass(createEliminateDeadSymbols());
    pm.addPass(createApplyInliner());
    pm.addPass(createInlineParametric(inlinerOpts));
    pm.addPass(createVerifyParameters(
        VerifyParametersOptions{/*simplifyParameters=*/true}));
    pm.addNestedPass<GeneratorOp>(createSROA());
    pm.addNestedPass<GeneratorOp>(createMem2Reg());
    pm.addNestedPass<GeneratorOp>(createCanonicalizer());
  }
}

void KGEN::buildElaborateModulePipeline(
    mlir::PassManager &pm, TargetInfoAttr target,
    const CompilationOptions &options, ElaboratorCompileAsmFn compileAsmFn,
    PackageGenLibraryFn packageGenLibraryFn) {
  pm.addPass(createEliminateDeadSymbols());
  // At the end of the LIT lowering pipeline, pull in the bodies of constructs
  // that were already elaborated.
  pm.addPass(createMaterializePackages(std::move(packageGenLibraryFn)));

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
  pm.addPass(createVerifyParameters());
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
