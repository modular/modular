//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h"
#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h"
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/Driver/Common/Utils.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/ConstantFoldPass.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/Passes.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h"
#include "GraphCompiler/Support/BlobLifetimeManager.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MLIRToMEF.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Support/Compiler/PassTimingUtils.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/Timing.h"

using namespace M;
using namespace MO;

void M::MO::loadDialects(mlir::DialectRegistry &registry) {
  registry.insert<MO::MODialect, MOSH::MOSHDialect, MOGG::MOGGDialect,
                  MGP::MGPDialect, GRT::GRTDialect, RMO::RMODialect,
                  MAP::MAPDialect>();
  M::registerAllKGENDialects(registry);
  M::registerKGENToLLVMTranslation(registry);
}

void M::MO::registerShapeInferencePipeline() {
  mlir::PassPipelineRegistration<>(
      "mo-shape-inference", "Run the end-to-end shape inference pipeline.",
      buildShapeInferencePipeline);
}

void M::MO::buildJitPipeline(mlir::OpPassManager &pm) {
  pm.addPass(MOGG::createLowerCompiledRoutines());
  pm.addPass(MGP::createInlineMGPCallsPass());
  pm.addPass(MGP::createFixupOverhangsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createOutlineModelsToCallables());
}

namespace {
static void gcPipelineImpl(mlir::OpPassManager &pm,
                           const GCPipelineOptions &opts, bool skipJit,
                           bool dumpIntermediates = false);
} // namespace

void M::MO::registerGCPipeline() {
  mlir::PassPipelineRegistration<GCPipelineOptions>(
      "gc-pipeline", "Full graph-compiler pipeline (MO → MGP → JIT).",
      [](mlir::OpPassManager &pm, const GCPipelineOptions &opts) {
        // Enable dumps here so buildGCPipeline stays dump-free for cache keys.
        gcPipelineImpl(pm, opts, /*skipJit=*/false,
                       /*dumpIntermediates=*/!getMaxIRDumpDir().empty());
      });
}

void M::MO::registerGCPipelineThroughEmitMojo() {
  mlir::PassPipelineRegistration<GCPipelineOptions>(
      "gc-pipeline-pre-jit",
      "Graph compiler through emit-mojo (MO → MGP → emit-mojo; no JIT).",
      [](mlir::OpPassManager &pm, const GCPipelineOptions &opts) {
        gcPipelineImpl(pm, opts, /*skipJit=*/true,
                       /*dumpIntermediates=*/!getMaxIRDumpDir().empty());
      });
}

void M::MO::registerMoPipeline() {
  mlir::PassPipelineRegistration<MoPipelineOptions>(
      "mo-pipeline",
      "Passes from graph input up to (but not including) mo-to-mogg.",
      [](mlir::OpPassManager &pm, const MoPipelineOptions &opts) {
        buildMoPipeline(pm, opts);
      });
}

void M::MO::registerMapPipeline() {
  mlir::PassPipelineRegistration<MapPipelineOptions>(
      "map-pipeline",
      "Passes between mo-to-map and mogg-to-mgp (both excluded).",
      [](mlir::OpPassManager &pm, const MapPipelineOptions &opts) {
        buildMapPipeline(pm, opts);
      });
}

void M::MO::registerMoggPipeline() {
  mlir::PassPipelineRegistration<MoggPipelineOptions>(
      "mogg-pipeline",
      "Passes between mo-to-mogg and mogg-to-mgp (both excluded).",
      [](mlir::OpPassManager &pm, const MoggPipelineOptions &opts) {
        buildMoggPipeline(pm, opts);
      });
}

void M::MO::registerMgpPipeline() {
  mlir::PassPipelineRegistration<MgpPipelineOptions>(
      "mgp-pipeline", "Passes between mogg-to-mgp and jit (both excluded).",
      [](mlir::OpPassManager &pm, const MgpPipelineOptions &opts) {
        buildMgpPipeline(pm, opts);
      });
}

void M::MO::registerJitPipeline() {
  mlir::PassPipelineRegistration<>(
      "jit-pipeline", "Passes that run after the jit pass.",
      [](mlir::OpPassManager &pm) { buildJitPipeline(pm); });
}

void M::MO::registerPasses() {
  MO::registerShapeInferencePipeline();
  MO::registerGCPipeline();
  MO::registerGCPipelineThroughEmitMojo();
  MO::registerMoPipeline();
  MO::registerMapPipeline();
  MO::registerMoggPipeline();
  MO::registerMgpPipeline();
  MO::registerJitPipeline();
  MO::registerMOToMAPPasses();
  MO::registerMOTransformsPasses();
  MAP::registerMAPTransformsPasses();
  MAP::registerMAPToMOGGPasses();
  MOGG::registerOutlineRoutinesPass();
  MGP::registerMOGGToMGP();

  MOGG::registerCanonicalizeFuncsForMerge();
  MOGG::registerMergeIdenticalFuncs();
  MOGG::registerAddProfileScopeTracingPass();
  MOGG::registerWrapFuncsWithTracePass();
  MOGG::registerAddTryExceptBlocksPass();
  MOGG::registerLookupConstantResources();
  MGP::registerRaiseBackIndexConstant();
  MOGG::registerOutlineAsyncRoutinesPass();
  MOGG::registerEmitMojo();
  MOGG::registerDumpMojo();
  MOGG::registerLoadMojo();
  MOGG::registerLowerCompiledRoutines();
  MGP::registerLowerPrimitives();
  MOGG::registerSequenceFusion();

  MGP::registerMGPTransformsPasses();
  MOGG::registerPasses();

  RMO::registerLegalizeRMOOpsPass();

  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerReconcileUnrealizedCastsPassPass();
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();
  mlir::registerViewOpGraphPass();

  KGEN::registerVerifyParametersPass();
}

LogicalResult MO::runGcPipeline(ModuleOp module,
                                const GCPipelineOptions &gcPipelineOptions) {

  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);

  if (!gcPipelineOptions.mlirTimingOutDir.empty()) {
    const char *pipelineName = "GC Pipeline";
    auto enabledTiming = M::configureMLIRPassTimingJSONOutput(
        pm, gcPipelineOptions.mlirTimingOutDir.getValue(), pipelineName);
    if (enabledTiming.isError()) {
      llvm::errs() << enabledTiming.getError() << "\n";
      return failure();
    }
  }

  MO::buildGCPipeline(pm, gcPipelineOptions);
  return pm.run(module);
}

void M::MO::buildMoPipeline(mlir::OpPassManager &pm,
                            const MoPipelineOptions &opts) {
  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  // Skipped when the caller already seeded decls and
  // `mogg.opaque_type_mapping` via `MOGG::copyKernelDecls`. RewriteOpaqueTypes
  // still runs so freshly-cloned `!mo.opaque<"simple">` uses get rewritten.
  if (!opts.kernelDeclsPrePopulated) {
    pm.addPass(MOGG::createImportPackages(
        {.mojoPkgPaths = kernelLibraryPaths, .litMlirPaths = {}}));
    pm.addPass(MOGG::createVerifyKernels());
  }

  pm.addPass(MOGG::createRewriteOpaqueTypes());
  pm.addPass(KGEN::createVerifyParameters());

  if (opts.debugPrintAllTensors)
    pm.addPass(MO::createDebugPrintAllTensorsPass());

  pm.addPass(RMO::createLegalizeRMOOps());

  pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createCanonicalizeChainSequences());

  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  std::string_view compiledFrameworkLabel;
  if (opts.compiledFrameworkLabel.hasValue())
    compiledFrameworkLabel = opts.compiledFrameworkLabel;

  std::optional<std::string> cacheDir;
  if (opts.cacheDir.hasValue())
    cacheDir = opts.cacheDir;

  if (opts.useShapeInference)
    buildShapeInferencePipeline(pm);

  // TODO(GEX-886): Remove symbolic optimization's dependency on constant
  // folding.
  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

  pm.addPass(MO::createHoistConstantSubgraphsIntoParent());
  pm.addNestedPass<MO::GraphOp>(mlir::createCSEPass());

  SymbolicOptimizationOptions symbolicOptimizationOpts;
  symbolicOptimizationOpts.symbolicFfastMath = opts.symbolicFfastMath;
  symbolicOptimizationOpts.symbolicNoUndefinedMath =
      opts.symbolicNoUndefinedMath;

  pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  pm.addPass(MO::createMergeDuplicateShapeMaterializations());
  pm.addPass(MO::createAddFallbackShapeFunctions());
  pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

  pm.addPass(MO::createInferLayouts());
  if (opts.usePatterns) {
    MOGG::PatternFusionOptions patternFusionOpts;
    patternFusionOpts.mhaAllowUnsafeBCast =
        opts.patternFusionMHAAllowUnsafeBCasts;
    patternFusionOpts.mhaAllowInternalReshape =
        opts.patternFusionMHAAllowInternalReshape;
    patternFusionOpts.enableMHAPattern = opts.patternFusionMHAEnable;
    patternFusionOpts.targetDevice = opts.defaultDeviceLabel;
    patternFusionOpts.targetDeviceId = opts.defaultDeviceId;
    pm.addPass(MOGG::createPatternFusion(patternFusionOpts));

    InvokeShapePackingFunctionsOptions shapeFuncOptions;
    shapeFuncOptions.sanityCheckShapeFuncs = opts.sanityCheckShapeFuncs;

    pm.addPass(MO::createInvokeShapePackingFunctions(shapeFuncOptions));
    pm.addPass(MO::createPropagateShapes());
    pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  }

  MO::ValidateDevicesPassOptions validateDevicesPassOpt;
  validateDevicesPassOpt.defaultDeviceTargetTriple =
      opts.defaultDeviceTargetTriple;
  validateDevicesPassOpt.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
  pm.addPass(MO::createValidateDevicesPass(validateDevicesPassOpt));

  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  // Absorb per-device consumers of an mo.tensor.unbundle back into the
  // producing mo.parallel (canonicalize collapses bundle(unbundle(X)) to X) so
  // epilogue fusion sees a single kernel body.
  pm.addPass(MO::createCreateParallelFromUnbundle());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createMergeAdjacentParallels());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

  if (opts.fuseJitKernels) {
    // Without fusion this can add extraneous copies.
    pm.addPass(MO::createElideRedundantMutableOps());
  }

  pm.addPass(MO::createHoistParameterExpressions());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createRemoveUnusedOps());
}

void M::MO::buildMapPipeline(mlir::OpPassManager &pm,
                             const MapPipelineOptions &opts) {
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  pm.addNestedPass<GraphOp>(MAP::createFuseKernels());
  if (opts.fuseMutableLoadStore)
    pm.addNestedPass<GraphOp>(MAP::createFuseMutableLoads());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  // After fusion settles, give every fused-output epilogue a real store pointer
  // (plus a pre-cast accumulation buffer for mixed-precision accumulation).
  pm.addNestedPass<GraphOp>(MAP::createAllocateAccumulatingBuffers());

  // An unfused reshape has no MAPToMOGG lowering (its hoisted stride op isn't
  // LowerableToMOGGInterface yet), and an unfused contiguous slice can take the
  // zero-copy path. Raise both back before outlining commits to MAP.
  pm.addNestedPass<GraphOp>(MAP::createRaiseBackContiguousViews());

  pm.addPass(MAP::createOutlineStructuredKernelBodies());
  // Uniques `map.iter.opaque` inputs and captures so the lowered kernel's
  // parameter list is minimal.
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(MAP::createLoopInvariantViewMotion());
  // CSE before canonicalize: two loads hoisting the same view create identical
  // stride ops; CSE must unify them so DedupeCapturingClosureCaptures can fold
  // the duplicate captures.
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
}

void M::MO::buildMoggPipeline(mlir::OpPassManager &pm,
                              const MoggPipelineOptions &) {
  pm.addPass(MOGG::createLegalizeRowwiseKernels());
  pm.addPass(MOGG::createConformCallOperands());
  pm.addPass(MOGG::createOutlineRoutines());
  pm.addPass(MOGG::createWrapFuncsWithTrace());

  // On the module, not GraphOp: the outlined stubs are its siblings.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void M::MO::buildMgpPipeline(mlir::OpPassManager &pm,
                             const MgpPipelineOptions &opts) {
  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(MGP::createConcatInPlacePass());
  pm.addPass(MGP::createExecInvariantCodeMotionPass());

  // After ExecInvariantCodeMotion: buffer reuse can block hoisting.
  pm.addPass(MGP::createSimpleBufferReusePass());

  pm.addPass(MGP::createDynamicMemoryPlanningPass());
  pm.addPass(MGP::createDeferAllocsPass());
  pm.addPass(MGP::createVerifyModelArgumentsPass());
  pm.addPass(MGP::createPredicateAssertDependenciesPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  // Before any fusion.
  pm.addPass(MGP::createAttachPythonStackTraces());

  pm.addPass(MGP::createGlobalizeSlotIndicesPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  MGP::LowerPrimitivesOptions lowerPrimOpts;
  lowerPrimOpts.kernelLibraryPaths = kernelLibraryPaths;
  lowerPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  pm.addPass(MGP::createLowerPrimitives(lowerPrimOpts));

  {
    pm.addPass(MGP::createStageDeviceGraphsPass());
    pm.addPass(MGP::createPartitionDeviceGraphCollectivePass());
    pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());
    pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
    pm.addPass(MGP::createSimplifyDeviceGraphDependenciesPass());
    pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());
    pm.addPass(MGP::createGenerateBuildGraphStubsPass());
    // Drop the dead launch context chain before finalization audits the
    // builder bridge's live users.
    pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
    pm.addPass(MGP::createFinalizeDeviceGraphsPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }

  pm.addPass(MOGG::createSequenceFusion());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  pm.addPass(MGP::createSimplifyExtendLifetimesPass());

  pm.addPass(MOGG::createOutlineAsyncRoutines());
  pm.addPass(MOGG::createLookupConstantResources());

  // Lifting slot-index constants into stub arguments lets kernels differing
  // only in those indices merge below.
  pm.addPass(MOGG::createLiftBufferCacheSlots());

  // The emitted Mojo reads neither device ids nor symbolic dim names, so
  // erasing them lets per-device and per-layer copies merge below.
  pm.addPass(MOGG::createCanonicalizeFuncsForMerge());

  // Stubs that differ only by a redundant op merge once it is gone.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MOGG::createMergeIdenticalFuncs());

  // index.constant emits fewer lines of Mojo than its lowered form.
  pm.addPass(MGP::createRaiseBackIndexConstant());

  pm.addPass(MGP::createFixupOverhangsPass());
  pm.addNestedPass<MOGG::FuncOp>(mlir::createCanonicalizerPass());

  // Drop stubs orphaned by MergeIdenticalFuncs.
  pm.addPass(mlir::createSymbolDCEPass());

  // Before try/except wrapping: the begin/end Trace calls raise, and the
  // MGP-ABI entry point must not.
  if (opts.enableProfileScopeTracing) {
    MOGG::AddProfileScopeTracingOptions profileScopeOptions;
    profileScopeOptions.profileScopeTracingTraceUnlabelled =
        opts.profileScopeTracingTraceUnlabelled;
    pm.addPass(MOGG::createAddProfileScopeTracing(profileScopeOptions));
  }

  // Last, so it sees the final raising set; try/except around non-raising
  // code warns in the Mojo compiler.
  pm.addPass(MOGG::createAddTryExceptBlocks());

  MOGG::EmitMojoOptions emitMojoOptions;
  emitMojoOptions.kernelLibraryPaths = kernelLibraryPaths;
  emitMojoOptions.enableOpTraceComments = opts.enableOpTraceComments;
  emitMojoOptions.enableComputeLambdas = opts.enableComputeLambdas;
  emitMojoOptions.moggDebugDumpFusedTensorSpecs =
      opts.moggDebugDumpFusedTensorSpecs;
  pm.addPass(MOGG::createEmitMojo(emitMojoOptions));

  if (opts.dumpMojoText)
    pm.addPass(MOGG::createDumpMojo());
}

namespace {

/// Appends every gc-pipeline stage to `pm`. With `dumpIntermediates`, a
/// DumpStagePass runs before the first stage and after each one.
static void gcPipelineImpl(mlir::OpPassManager &pm,
                           const GCPipelineOptions &opts, bool skipJit,
                           bool dumpIntermediates) {

  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  // Filled by the first ("mo") dump from the non-subgraph model names while
  // the GraphOps still exist; later dumps reuse it.
  auto dumpBaseName = std::make_shared<std::string>();

  if (dumpIntermediates)
    pm.addPass(MO::createDumpStagePass("mo", dumpBaseName));

  MoPipelineOptions moOpts;
  moOpts.debugPrintAllTensors = opts.debugPrintAllTensors;
  moOpts.useShapeInference = opts.useShapeInference;
  moOpts.constantFold = opts.constantFold;
  if (opts.compiledFrameworkLabel.hasValue())
    moOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel.getValue();
  if (opts.cacheDir.hasValue())
    moOpts.cacheDir = opts.cacheDir.getValue();
  moOpts.symbolicFfastMath = opts.symbolicFfastMath;
  moOpts.symbolicNoUndefinedMath = opts.symbolicNoUndefinedMath;
  moOpts.kernelLibraryPaths = kernelLibraryPaths;
  moOpts.kernelDeclsPrePopulated = opts.kernelDeclsPrePopulated;
  moOpts.usePatterns = opts.usePatterns;
  moOpts.patternFusionMHAEnable = opts.patternFusionMHAEnable;
  moOpts.patternFusionMHAAllowUnsafeBCasts =
      opts.patternFusionMHAAllowUnsafeBCasts;
  moOpts.patternFusionMHAAllowInternalReshape =
      opts.patternFusionMHAAllowInternalReshape;
  moOpts.sanityCheckShapeFuncs = opts.sanityCheckShapeFuncs;
  moOpts.defaultDeviceLabel = opts.defaultDeviceLabel;
  moOpts.defaultDeviceId = opts.defaultDeviceId;
  moOpts.defaultDeviceTargetTriple = opts.defaultDeviceTargetTriple;
  moOpts.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
  moOpts.fuseJitKernels = opts.fuseJitKernels;

  MapPipelineOptions mapOpts;
  mapOpts.fuseMutableLoadStore = opts.fuseMutableLoadStore;

  MoggPipelineOptions moggOpts;

  MGP::MOGGToMGPOptions moToPrimOpts;
  moToPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  if (opts.compiledFrameworkLabel.hasValue())
    moToPrimOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;

  MgpPipelineOptions mgpOpts;
  mgpOpts.kernelLibraryPaths = kernelLibraryPaths;
  mgpOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  mgpOpts.enableOpTraceComments = opts.enableOpTraceComments;
  mgpOpts.enableComputeLambdas = opts.enableComputeLambdas;
  mgpOpts.moggDebugDumpFusedTensorSpecs = opts.moggDebugDumpFusedTensorSpecs;
  mgpOpts.dumpMojoText = opts.dumpMojoText;
  mgpOpts.enableProfileScopeTracing = opts.enableProfileScopeTracing;
  mgpOpts.profileScopeTracingTraceUnlabelled =
      opts.profileScopeTracingTraceUnlabelled;

  MOGG::JitCompileKernelsOptions jitOpts;
  if (!skipJit) {
    jitOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
    jitOpts.saveTempPrefix = opts.saveTempPrefix;
    jitOpts.createMOGGReproducers = opts.createMOGGReproducers;
    jitOpts.kernelLibraryPaths = kernelLibraryPaths;
    jitOpts.timeMOGGCompile = opts.timeMOGGCompile;
    jitOpts.debugKGENStubs = opts.debugKGENStubs;
    // Cross-compile for the requested GPU instead of probing the host.
    jitOpts.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
    jitOpts.defaultDeviceTargetTriple = opts.defaultDeviceTargetTriple;
    jitOpts.compileForHostExecution = opts.compileForHostExecution;
  }

  auto buildStage = [&](auto &&build, const std::string &dumpSuffix) {
    build(pm);
    if (dumpIntermediates && !dumpSuffix.empty())
      pm.addPass(MO::createDumpStagePass(dumpSuffix, dumpBaseName));
  };

  buildStage([&](auto &p) { buildMoPipeline(p, moOpts); }, "mo-pre-map");

  // Stable-tensor marking must see MO ops, so it runs as the last MO-level
  // pass before the dialect is lowered away.
  buildStage(
      [&](auto &p) {
        p.addPass(MO::createMarkStableTensorsPass());
        p.addPass(MO::createMOToMAP());
      },
      "mo-to-map");

  buildStage(
      [&](auto &p) {
        buildMapPipeline(p, mapOpts);
        // After MAP fusion so a check never fuses into the kernel it checks;
        // before map-to-mogg so the injected checks compile with the rest.
        if (opts.nanCheck)
          p.addPass(MO::createNanCheckPass(opts.nanCheckStride));
      },
      "map-pre-mogg");

  MAP::MAPToMOGGOptions mapToMoggOpts;
  mapToMoggOpts.enableComputeLambdas = opts.enableComputeLambdas;
  buildStage([&](auto &p) { p.addPass(MAP::createMAPToMOGG(mapToMoggOpts)); },
             "map-to-mogg");

  buildStage(
      [&](auto &p) {
        buildMoggPipeline(p, moggOpts);
        // OpaqueLoc tracebacks can't be printed, so convert them for the
        // dump. Dump-only: this strips the OpaqueLocs buildMgpPipeline's
        // assert handling reads.
        if (dumpIntermediates)
          p.addPass(MGP::createAttachPythonStackTraces());
      },
      "mogg-pre-mgp");

  buildStage([&](auto &p) { p.addPass(MGP::createMOGGToMGP(moToPrimOpts)); },
             "mogg-to-mgp");

  buildStage(
      [&](auto &p) {
        buildMgpPipeline(p, mgpOpts);
        if (!skipJit) {
          p.addPass(MOGG::createJitCompileKernels(jitOpts));
          p.addPass(MOGG::createStripKernelDecls());
          buildJitPipeline(p);
        }
      },
      "mgp");
}

} // namespace

void M::MO::buildGCPipeline(mlir::OpPassManager &pm,
                            const GCPipelineOptions &opts) {
  gcPipelineImpl(pm, opts, /*skipJit=*/false);
}

void M::MO::buildGCPipelineThroughEmitMojo(mlir::OpPassManager &pm,
                                           const GCPipelineOptions &opts) {
  gcPipelineImpl(pm, opts, /*skipJit=*/true);
}

LogicalResult M::MO::runGCPipeline(ModuleOp moduleOp,
                                   const GCPipelineOptions &opts,
                                   bool dumpIntermediates) {
  auto pm = mlir::PassManager::on<ModuleOp>(
      moduleOp.getContext(), mlir::OpPassManager::Nesting::Implicit);

  if (!opts.mlirTimingOutDir.empty()) {
    auto enabledTiming = M::configureMLIRPassTimingJSONOutput(
        pm, opts.mlirTimingOutDir.getValue(), "GC Pipeline");
    if (enabledTiming.isError()) {
      llvm::errs() << enabledTiming.getError() << "\n";
      return failure();
    }
  }

  gcPipelineImpl(pm, opts, /*skipJit=*/opts.preJit, dumpIntermediates);
  return pm.run(moduleOp);
}

void M::MO::buildShapeInferencePipeline(mlir::OpPassManager &pm) {
  // Turns static reshapes like `[D0] -> [1]` into rebinds (so staticization
  // needn't look through them) and materializes parameter definitions for
  // `mo.rebind`s that still declare output parameters.
  pm.addNestedPass<MO::GraphOp>(mlir::createCanonicalizerPass());

  // Evaluates constant ops and shape functions in the parameter domain, then
  // simplifies parameter bindings by substitution and constraint solving.
  pm.addNestedPass<MO::GraphOp>(MO::createPropagateShapes());
  pm.addNestedPass<MO::GraphOp>(mlir::createCSEPass());
  pm.addNestedPass<MO::GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(KGEN::createVerifyParameters());
}

using BlobMap = DenseMap<const uint8_t *, mlir::AsmResourceBlob *>;

static BlobMap getDenseResources(ModuleOp module) {
  BlobMap result;
  for (Operation &op : module.getOps()) {
    if (auto modelOp = dyn_cast<MGP::ModelOp>(op)) {
      modelOp.walk([&](MGP::BufferConstantOp constOp) {
        auto elementsAttr = constOp.getValue();
        if (auto resourceAttr =
                dyn_cast<DenseResourceElementsAttr>(elementsAttr)) {
          mlir::AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob();
          if (blob->getDeleter())
            result[blob->getDataAs<uint8_t>().data()] = blob;
        }
      });
    }
  }

  return result;
}

ErrorOr<BufferRef> M::MO::convertMGPToMEF(ModuleOp module,
                                          bool enableMEFMetadata) {
  OwningOpRef<ModuleOp> moduleToRewrite = module.clone();
  // Device-generic ops get renamed with their device label so GraphRT can
  // dispatch to a device-appropriate primitive.
  BlobMap resources = getDenseResources(*moduleToRewrite);
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module->getContext(), mlir::OpPassManager::Nesting::Implicit);
  pm.addPass(MGP::createPrepareDevicePrimitivesPass());
  FormatScopedDiagnosticHandler diagnostics(moduleToRewrite->getContext());
  if (failed(pm.run(*moduleToRewrite)))
    return Error(Twine("Failed to prepare MGP model for conversion to MEF. ") +
                 diagnostics.formatMessage());
  ConcatenationTree result =
      translateMLIRToMEF(*moduleToRewrite, enableMEFMetadata);
  if (result.getSize() == 0) {
    return Error(Twine("Failed to convert model to MEF. ") +
                 diagnostics.formatMessage());
  }
  M::WriteableBufferRef outputBuffer = WriteableBuffer::get(
      0, maxAttributeAlignment().value(), result.getSize());
  result.traverse([&](ArrayRef<uint8_t> data) {
    outputBuffer->write(reinterpret_cast<const char *>(data.data()),
                        data.size());
    if (resources.count(data.data())) {
      if (auto blob = resources[data.data()]; blob->getDeleter()) {
        // Free owned blob data as soon as it's been copied out.
        auto ref = M::GML::BlobLifetimeManager(std::move(blob->getDeleter()),
                                               blob->getData(),
                                               blob->getDataAlignment());
        blob->getDeleter() = nullptr;
      }
    }
  });

  return outputBuffer;
}
