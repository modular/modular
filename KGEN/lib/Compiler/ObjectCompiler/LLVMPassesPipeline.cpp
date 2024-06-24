//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMPassesPipeline.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Process.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#include "llvm/Transforms/Coroutines/CoroConditionalWrapper.h"
#include "llvm/Transforms/Coroutines/CoroEarly.h"
#include "llvm/Transforms/Coroutines/CoroElide.h"
#include "llvm/Transforms/Coroutines/CoroSplit.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Annotation2Metadata.h"
#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/Transforms/IPO/CalledValuePropagation.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/ElimAvailExtern.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include "llvm/Transforms/Instrumentation/CGProfile.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/AlignmentFromAssumptions.h"
#include "llvm/Transforms/Scalar/AnnotationRemarks.h"
#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/Transforms/Scalar/CallSiteSplitting.h"
#include "llvm/Transforms/Scalar/ConstraintElimination.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/DivRemPairs.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopDistribute.h"
#include "llvm/Transforms/Scalar/LoopFlatten.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/LoopInstSimplify.h"
#include "llvm/Transforms/Scalar/LoopInterchange.h"
#include "llvm/Transforms/Scalar/LoopLoadElimination.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopSink.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/SpeculativeExecution.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Scalar/WarnMissedTransforms.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/LibCallsShrinkWrap.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/RelLookupTableConverter.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "llvm/Transforms/Vectorize/VectorCombine.h"

using namespace llvm;
using namespace M::KGEN;

static bool isGPUBackend(const CompilationOptions &options) {
  llvm::Triple triple(options.targetTriple);
  return llvm::is_contained({llvm::Triple::nvptx, llvm::Triple::nvptx64},
                            triple.getArch());
}

static SimplifyCFGOptions
adjustSimplifyCFGOptions(SimplifyCFGOptions simplifyCFGOptions,
                         const CompilationOptions &options) {
  if (!isGPUBackend(options))
    return simplifyCFGOptions;

  // On GPUs the branch cost is much larger than that of the CPU, so increase
  // the threshold. E.g. if we have
  //
  // if (cond0) return
  // if (cond1) return
  // <<stuff>>
  //
  // then we want to simplify this to
  //
  // if (cond0 | cond1) return
  // <<stuff>>
  return simplifyCFGOptions.bonusInstThreshold(2);
}

static void addSanitizers(ModulePassManager &modulePassManager,
                          const CompilationOptions &options) {
  // LLVM's sanitizer instrumentation is not supported for NVPTX.
  if (isGPUBackend(options))
    return;

  if (options.sanitizers.has(M::Sanitizers::kThread)) {
    modulePassManager.addPass(ModuleThreadSanitizerPass());
    modulePassManager.addPass(
        llvm::createModuleToFunctionPassAdaptor(llvm::ThreadSanitizerPass()));
  }

  if (options.sanitizers.has(M::Sanitizers::kAddress)) {
    AddressSanitizerOptions Opts;
    bool moduleUseAfterScope = false;
    bool useOdrIndicator = false;
    modulePassManager.addPass(
        AddressSanitizerPass(Opts, moduleUseAfterScope, useOdrIndicator));
  }
}

static FunctionPassManager
buildFunctionSimplificationPipeline(const CompilationOptions &options) {
  FunctionPassManager FPM;
  // Form SSA out of local memory accesses after breaking apart aggregates into
  // scalars.
  FPM.addPass(SROAPass(SROAOptions::ModifyCFG));

  // Catch trivial redundancies
  FPM.addPass(EarlyCSEPass(true /* Enable mem-ssa. */));

  // Speculative execution if the target has divergent branches; otherwise nop.
  FPM.addPass(SpeculativeExecutionPass(/* OnlyIfDivergentTarget =*/true));

  // Optimize based on known information about branches, and cleanup afterward.
  FPM.addPass(JumpThreadingPass());
  FPM.addPass(CorrelatedValuePropagationPass());

  SimplifyCFGOptions simplifyCFGOptions = adjustSimplifyCFGOptions(
      SimplifyCFGOptions().convertSwitchRangeToICmp(true), options);

  FPM.addPass(SimplifyCFGPass(simplifyCFGOptions));
  FPM.addPass(InstCombinePass());
  FPM.addPass(AggressiveInstCombinePass());

  FPM.addPass(ConstraintEliminationPass());

  FPM.addPass(LibCallsShrinkWrapPass());

  FPM.addPass(TailCallElimPass());
  FPM.addPass(SimplifyCFGPass(simplifyCFGOptions));

  // Form canonically associated expression trees, and simplify the trees using
  // basic mathematical properties. For example, this will form (nearly)
  // minimal multiplication trees.
  FPM.addPass(ReassociatePass());

  // Add the primary loop simplification pipeline.
  // FIXME: Currently this is split into two loop pass pipelines because we run
  // some function passes in between them. These can and should be removed
  // and/or replaced by scheduling the loop pass equivalents in the correct
  // positions. But those equivalent passes aren't powerful enough yet.
  // Specifically, `SimplifyCFGPass` and `InstCombinePass` are currently still
  // used. We have `LoopSimplifyCFGPass` which isn't yet powerful enough yet to
  // fully replace `SimplifyCFGPass`, and the closest to the other we have is
  // `LoopInstSimplify`.
  LoopPassManager LPM1, LPM2;

  // Simplify the loop body. We do this initially to clean up after other loop
  // passes run, either when iterating on a loop or on inner loops with
  // implications on the outer loop.
  LPM1.addPass(LoopInstSimplifyPass());
  LPM1.addPass(LoopSimplifyCFGPass());

  // Try to remove as much code from the loop header as possible,
  // to reduce amount of IR that will have to be duplicated. However,
  // do not perform speculative hoisting the first time as LICM
  // will destroy metadata that may not need to be destroyed if run
  // after loop rotation.
  // TODO: Investigate promotion cap for O1.
  LPM1.addPass(LICMPass(/*LicmMssaOptCap*/ 100,
                        /*LicmMssaNoAccForPromotionCap*/ 250,
                        /*AllowSpeculation=*/false));

  // Disable header duplication in loop rotation at -Oz.
  LPM1.addPass(LoopRotatePass(/*EnableHeaderDuplication*/ true, false));
  // TODO: Investigate promotion cap for O1.
  LPM1.addPass(LICMPass(/*LicmMssaOptCap*/ 100,
                        /*LicmMssaNoAccForPromotionCap*/ 250,
                        /*AllowSpeculation=*/true));
  LPM1.addPass(SimpleLoopUnswitchPass(/* NonTrivial */ true));

  LPM2.addPass(LoopIdiomRecognizePass());
  LPM2.addPass(IndVarSimplifyPass());

  LPM2.addPass(LoopDeletionPass());

  FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM1),
                                              /*UseMemorySSA=*/true,
                                              /*UseBlockFrequencyInfo=*/true));

  FPM.addPass(SimplifyCFGPass(simplifyCFGOptions));
  FPM.addPass(InstCombinePass());
  // The loop passes in LPM2 (LoopIdiomRecognizePass, IndVarSimplifyPass,
  // LoopDeletionPass and LoopFullUnrollPass) do not preserve MemorySSA.
  // *All* loop passes must preserve it, in order to be able to use it.
  FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM2),
                                              /*UseMemorySSA=*/false,
                                              /*UseBlockFrequencyInfo=*/false));

  // Delete small array after loop unroll.
  FPM.addPass(SROAPass(SROAOptions::ModifyCFG));

  // Try vectorization/scalarization transforms that are both improvements
  // themselves and can allow further folds with GVN and InstCombine.
  FPM.addPass(VectorCombinePass(/*TryEarlyFoldsOnly=*/true));

  // Eliminate redundancies.
  FPM.addPass(MergedLoadStoreMotionPass());
  FPM.addPass(GVNPass());

  // Sparse conditional constant propagation.
  // FIXME: It isn't clear why we do this *after* loop passes rather than
  // before...
  FPM.addPass(SCCPPass());

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  FPM.addPass(BDCEPass());

  // Run instcombine after redundancy and dead bit elimination to exploit
  // opportunities opened up by them.
  FPM.addPass(InstCombinePass());

  FPM.addPass(JumpThreadingPass());
  FPM.addPass(CorrelatedValuePropagationPass());

  // Finally, do an expensive DCE pass to catch all the dead code exposed by
  // the simplifications and basic cleanup after all the simplifications.
  // TODO: Investigate if this is too expensive.
  FPM.addPass(ADCEPass());

  // Specially optimize memory movement as it doesn't look like dataflow in SSA.
  FPM.addPass(MemCpyOptPass());

  FPM.addPass(DSEPass());
  FPM.addPass(createFunctionToLoopPassAdaptor(
      LICMPass(/*LicmMssaOptCap*/ 100, /*LicmMssaNoAccForPromotionCap*/ 250,
               /*AllowSpeculation=*/true),
      /*UseMemorySSA=*/true, /*UseBlockFrequencyInfo=*/true));

  FPM.addPass(CoroElidePass());

  FPM.addPass(SimplifyCFGPass(
      adjustSimplifyCFGOptions(SimplifyCFGOptions()
                                   .convertSwitchRangeToICmp(true)
                                   .hoistCommonInsts(true)
                                   .sinkCommonInsts(true),
                               options)));
  FPM.addPass(InstCombinePass());

  return FPM;
}

static void addInlinerPasses(ModulePassManager &MPM,
                             const CompilationOptions &options) {
  ModuleInlinerWrapperPass MIWP(
      getInlineParams(/*speed*/ 3, /*size*/ 0),
      /*PerformMandatoryInliningsFirst*/ true,
      InlineContext{ThinOrFullLTOPhase::None, InlinePass::CGSCCInliner},
      InliningAdvisorMode::Default,
      /*MaxDevirtIterations*/ 4);

  // Require the GlobalsAA analysis for the module so we can query it within
  // the CGSCC pipeline.
  MIWP.addModulePass(RequireAnalysisPass<GlobalsAA, Module>());
  // Invalidate AAManager so it can be recreated and pick up the newly
  // available GlobalsAA.
  MIWP.addModulePass(
      createModuleToFunctionPassAdaptor(InvalidateAnalysisPass<AAManager>()));

  // Require the ProfileSummaryAnalysis for the module so we can query it
  // within the inliner pass.
  MIWP.addModulePass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());
  // Now begin the main postorder CGSCC pipeline.
  // FIXME: The current CGSCC pipeline has its origins in the legacy pass
  // manager and trying to emulate its precise behavior. Much of this doesn't
  // make a lot of sense and we should revisit the core CGSCC structure.
  CGSCCPassManager &MainCGPipeline = MIWP.getPM();

  // Note: historically, the PruneEH pass was run first to deduce nounwind and
  // generally clean up exception handling overhead. It isn't clear this is
  // valuable as the inliner doesn't currently care whether it is inlining an
  // invoke or a call.

  // Now deduce any function attributes based in the current code.
  MainCGPipeline.addPass(PostOrderFunctionAttrsPass());

  // Lastly, add the core function simplification pipeline nested inside the
  // CGSCC walk.
  MainCGPipeline.addPass(createCGSCCToFunctionPassAdaptor(
      buildFunctionSimplificationPipeline(options),
      /*EagerlyInvalidateAnalyses*/ true,
      /*EnableNoRerunSimplificationPipeline*/ true));

  MainCGPipeline.addPass(CoroSplitPass(true));
  MPM.addPass(std::move(MIWP));
}

static void addVectorPasses(FunctionPassManager &FPM,
                            const CompilationOptions &options) {
  // Eliminate loads by forwarding stores from the previous iteration to loads
  // of the current iteration.
  FPM.addPass(LoopLoadEliminationPass());

  // Cleanup after the loop optimization passes.
  FPM.addPass(InstCombinePass());

  // Now that we've formed fast to execute loop structures, we do further
  // optimizations. These are run afterward as they might block doing complex
  // analyses and transforms such as what are needed for loop vectorization.

  // Cleanup after loop vectorization, etc. Simplification passes like CVP and
  // GVN, loop transforms, and others have already run, so it's now better to
  // convert to more optimized IR using more aggressive simplify CFG options.
  // The extra sinking transform can create larger basic blocks, so do this
  // before SLP vectorization.
  FPM.addPass(SimplifyCFGPass(
      adjustSimplifyCFGOptions(SimplifyCFGOptions()
                                   .forwardSwitchCondToPhi(true)
                                   .convertSwitchRangeToICmp(true)
                                   .convertSwitchToLookupTable(true)
                                   .needCanonicalLoops(false)
                                   .hoistCommonInsts(true)
                                   .sinkCommonInsts(true),
                               options)));

  FPM.addPass(SLPVectorizerPass());
  // Enhance/cleanup vector code.
  FPM.addPass(VectorCombinePass());

  FPM.addPass(InstCombinePass());
  // Now that we are done with loop unrolling, be it either by LoopVectorizer,
  // or LoopUnroll passes, some variable-offset GEP's into alloca's could have
  // become constant-offset, thus enabling SROA and alloca promotion. Do so.
  // NOTE: we are very late in the pipeline, and we don't have any LICM
  // or SimplifyCFG passes scheduled after us, that would cleanup
  // the CFG mess this may created if allowed to modify CFG, so forbid that.
  FPM.addPass(SROAPass(SROAOptions::PreserveCFG));
  FPM.addPass(InstCombinePass());
  FPM.addPass(
      RequireAnalysisPass<OptimizationRemarkEmitterAnalysis, Function>());
  FPM.addPass(createFunctionToLoopPassAdaptor(
      LICMPass(/*LicmMssaOptCap*/ 100, /*LicmMssaNoAccForPromotionCap*/ 250,
               /*AllowSpeculation=*/true),
      /*UseMemorySSA=*/true, /*UseBlockFrequencyInfo=*/true));

  // Now that we've vectorized and unrolled loops, we may have more refined
  // alignment information, try to re-derive it here.
  FPM.addPass(AlignmentFromAssumptionsPass());
}

static ModulePassManager buildO3Pipeline(const CompilationOptions &options) {
  ModulePassManager MPM;

  // Do basic inference of function attributes from known properties of system
  // libraries and other oracles.
  MPM.addPass(InferFunctionAttrsPass());
  MPM.addPass(CoroEarlyPass());

  // Create an early function pass manager to cleanup the output of the
  // frontend.
  FunctionPassManager EarlyFPM;
  // Lower llvm.expect to metadata before attempting transforms.
  // Compare/branch metadata may alter the behavior of passes like SimplifyCFG.
  EarlyFPM.addPass(LowerExpectIntrinsicPass());
  EarlyFPM.addPass(SimplifyCFGPass());
  EarlyFPM.addPass(SROAPass(SROAOptions::ModifyCFG));
  EarlyFPM.addPass(EarlyCSEPass());
  EarlyFPM.addPass(CallSiteSplittingPass());

  MPM.addPass(
      createModuleToFunctionPassAdaptor(std::move(EarlyFPM),
                                        /*EagerlyInvalidateAnalyses*/ true));

  // Promote any localized globals to SSA registers.
  // FIXME: Should this instead by a run of SROA?
  // FIXME: We should probably run instcombine and simplifycfg afterward to
  // delete control flows that are dead once globals have been folded to
  // constants.
  MPM.addPass(createModuleToFunctionPassAdaptor(PromotePass()));

  // Create a small function pass pipeline to cleanup after all the global
  // optimizations.
  FunctionPassManager GlobalCleanupPM;
  GlobalCleanupPM.addPass(InstCombinePass());

  SimplifyCFGOptions simplifyCFGOptions = adjustSimplifyCFGOptions(
      SimplifyCFGOptions().convertSwitchRangeToICmp(true), options);

  GlobalCleanupPM.addPass(SimplifyCFGPass(simplifyCFGOptions));
  MPM.addPass(
      createModuleToFunctionPassAdaptor(std::move(GlobalCleanupPM),
                                        /*EagerlyInvalidateAnalyses*/ true));

  addInlinerPasses(MPM, options);

  MPM.addPass(CoroCleanupPass());

  // Optimize globals now that the module is fully simplified.
  MPM.addPass(GlobalDCEPass());

  // Do RPO function attribute inference across the module to forward-propagate
  // attributes where applicable.
  // FIXME: Is this really an optimization rather than a canonicalization?
  MPM.addPass(ReversePostOrderFunctionAttrsPass());

  // Re-compute GlobalsAA here prior to function passes. This is particularly
  // useful as the above will have inlined, DCE'ed, and function-attr
  // propagated everything. We should at this point have a reasonably minimal
  // and richly annotated call graph. By computing aliasing and mod/ref
  // information for all local globals here, the late loop passes and notably
  // the vectorizer will be able to use them to help recognize vectorizable
  // memory operations.
  MPM.addPass(RecomputeGlobalsAAPass());

  FunctionPassManager OptimizePM;
  OptimizePM.addPass(Float2IntPass());
  OptimizePM.addPass(LowerConstantIntrinsicsPass());

  // FIXME: We need to run some loop optimizations to re-rotate loops after
  // simplifycfg and others undo their rotation.

  // Optimize the loop execution. These passes operate on entire loop nests
  // rather than on each loop in an inside-out manner, and so they are actually
  // function passes.

  LoopPassManager LPM;
  // First rotate loops that may have been un-rotated by prior passes.
  // Disable header duplication at -Oz.
  LPM.addPass(
      LoopRotatePass(/*EnableHeaderDuplication*/ true, /*LTOPreLink*/ false));
  // Some loops may have become dead by now. Try to delete them.
  // FIXME: see discussion in https://reviews.llvm.org/D112851,
  //        this may need to be revisited once we run GVN before loop deletion
  //        in the simplification pipeline.
  LPM.addPass(LoopDeletionPass());
  OptimizePM.addPass(createFunctionToLoopPassAdaptor(
      std::move(LPM), /*UseMemorySSA=*/false, /*UseBlockFrequencyInfo=*/false));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependencies
  // into separate loop that would otherwise inhibit vectorization.  This is
  // currently only performed for loops marked with the metadata
  // llvm.loop.distribute=true or when -enable-loop-distribute is specified.
  OptimizePM.addPass(LoopDistributePass());

  // Populates the VFABI attribute with the scalar-to-vector mappings
  // from the TargetLibraryInfo.
  OptimizePM.addPass(InjectTLIMappings());

  addVectorPasses(OptimizePM, options);

  // LoopSink pass sinks instructions hoisted by LICM, which serves as a
  // canonicalization pass that enables other optimizations. As a result,
  // LoopSink pass needs to be a very late IR pass to avoid undoing LICM
  // result too early.
  OptimizePM.addPass(LoopSinkPass());

  // And finally clean up LCSSA form before generating code.
  OptimizePM.addPass(InstSimplifyPass());

  // This hoists/decomposes div/rem ops. It should run after other sink/hoist
  // passes to avoid re-sinking, but before SimplifyCFG because it can allow
  // flattening of blocks.
  OptimizePM.addPass(DivRemPairsPass());

  // Try to annotate calls that were created during optimization.
  OptimizePM.addPass(TailCallElimPass());

  // LoopSink (and other loop passes since the last simplifyCFG) might have
  // resulted in single-entry-single-exit or empty blocks. Clean up the CFG.

  OptimizePM.addPass(SimplifyCFGPass(simplifyCFGOptions));

  // Add the core optimizing pipeline.
  MPM.addPass(
      createModuleToFunctionPassAdaptor(std::move(OptimizePM),
                                        /*EagerlyInvalidateAnalyses*/ true));

  // Add any relevant sanitizers.
  addSanitizers(MPM, options);

  // Now we need to do some global optimization transforms.
  // FIXME: It would seem like these should come first in the optimization
  // pipeline and maybe be the bottom of the canonicalization pipeline? Weird
  // ordering here.
  MPM.addPass(GlobalDCEPass());
  MPM.addPass(ConstantMergePass());

  MPM.addPass(CGProfilePass(false));

  // TODO: Relative look table converter pass caused an issue when full lto is
  // enabled. See https://reviews.llvm.org/D94355 for more details.
  // Until the issue fixed, disable this pass during pre-linking phase.
  MPM.addPass(RelLookupTableConverterPass());

  // Emit annotation remarks.
  MPM.addPass(createModuleToFunctionPassAdaptor(AnnotationRemarksPass()));

  return MPM;
}

static ModulePassManager buildO0Pipeline(const CompilationOptions &options) {
  ModulePassManager MPM;

  // Build a minimal pipeline based on the semantics required by LLVM,
  // which is just that always inlining occurs. Further, disable generating
  // lifetime intrinsics to avoid enabling further optimizations during
  // code generation.
  MPM.addPass(AlwaysInlinerPass(
      /*InsertLifetimeIntrinsics=*/false));

  ModulePassManager CoroPM;
  CoroPM.addPass(CoroEarlyPass());

  CGSCCPassManager CGPM;
  CGPM.addPass(CoroSplitPass());
  CoroPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  CoroPM.addPass(CoroCleanupPass());
  CoroPM.addPass(GlobalDCEPass());
  MPM.addPass(CoroConditionalWrapper(std::move(CoroPM)));

  // Add any relevant sanitizers.
  addSanitizers(MPM, options);

  MPM.addPass(createModuleToFunctionPassAdaptor(AnnotationRemarksPass()));

  return MPM;
}

ModulePassManager
M::KGEN::buildLLVMOptimizationPipeline(const CompilationOptions &options) {
  CodeGenOptLevel optLevel = options.getCodeGenOptLevel();

  assert((optLevel == CodeGenOptLevel::None ||
          optLevel == CodeGenOptLevel::Aggressive) &&
         "only OptLevel::None and OptLevel::Aggressive are supported");
  if (optLevel == CodeGenOptLevel::None)
    return buildO0Pipeline(options);
  return buildO3Pipeline(options);
}

static TargetPassConfig *
buildPassesToGenerateNVPTXCode(LLVMTargetMachine &tm, PassManagerBase &pm,
                               bool disableVerify,
                               MachineModuleInfoWrapperPass &mmiwp) {
  // Targets may override createPassConfig to provide a target-specific
  // subclass.
  TargetPassConfig *passConfig = tm.createPassConfig(pm);
  if (!passConfig)
    return nullptr;

  // Set PassConfig options provided by TargetMachine.
  passConfig->setDisableVerify(disableVerify);
  pm.add(passConfig);
  pm.add(&mmiwp);

  if (passConfig->addISelPasses())
    return nullptr;

  // Disable MachineSink pass which causes undesirable instruction reordering.
  // This fixes MOCO-712, MOCO-790 and MOCO-803.
  passConfig->disablePass(&MachineSinkingID);

  passConfig->addMachinePasses();
  passConfig->setInitialized();
  return passConfig;
}

static bool buildNVPTXLLcPipeline(LLVMTargetMachine &targetMachine,
                                  llvm::legacy::PassManagerBase &pm,
                                  raw_pwrite_stream &out,
                                  raw_pwrite_stream *dwoOut,
                                  CodeGenFileType fileType, bool disableVerify,
                                  MachineModuleInfoWrapperPass *mmiwp) {
  TargetPassConfig *passConfig =
      buildPassesToGenerateNVPTXCode(targetMachine, pm, disableVerify, *mmiwp);

  if (!passConfig)
    return true;

  if (TargetPassConfig::willCompleteCodeGenPipeline()) {
    if (targetMachine.addAsmPrinter(pm, out, dwoOut, fileType,
                                    mmiwp->getMMI().getContext()))
      return true;
  } else {
    // MIR printing is redundant with -filetype=null.
    if (fileType != CodeGenFileType::Null)
      pm.add(createPrintMIRPass(out));
  }

  pm.add(createFreeMachineFunctionPass());
  return false;
}

bool M::KGEN::addPassesToEmitFile(CompilationOptions &options,
                                  LLVMTargetMachine &targetMachine,
                                  llvm::legacy::PassManagerBase &pm,
                                  raw_pwrite_stream &out,
                                  raw_pwrite_stream *dwoOut,
                                  CodeGenFileType fileType, bool disableVerify,
                                  MachineModuleInfoWrapperPass *mmiwp) {

  if (isGPUBackend(options)) {
    return buildNVPTXLLcPipeline(targetMachine, pm, out, dwoOut, fileType,
                                 disableVerify, mmiwp);
  }

  return targetMachine.addPassesToEmitFile(pm, out, dwoOut, fileType,
                                           disableVerify, mmiwp);
}
