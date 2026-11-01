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

#ifndef GENERICML_MO_GRAPHCOMPILER_PASSES_H
#define GENERICML_MO_GRAPHCOMPILER_PASSES_H

#include "AsyncRT/ForwardDecls.h"
#include "GraphCompiler/GraphCompiler.h"
#include "Support/AlignedAlloc.h"
#include "mlir/Pass/PassOptions.h"

namespace M::MO {

/// MLIR options for `buildGCPipeline` and `buildGCPipelineThroughEmitMojo`.
/// See GraphCompilerCLOptions for the mt equivalents.
struct GCPipelineOptions : public mlir::PassPipelineOptions<GCPipelineOptions> {
  Option<std::string> compiledFrameworkLabel{
      *this, "compiled-framework-label",
      llvm::cl::desc(
          "Framework label to compile to (used in mgp.model framework tag).")};
  Option<size_t> minimumCpuAlignment{
      *this, "minimum-cpu-alignment",
      llvm::cl::desc("Alignment for all CPU-allocated buffers."),
      llvm::cl::init(kPreferredMemoryAlignment)};
  Option<bool> symbolicFfastMath{
      *this, "symbolic-ffast-math",
      llvm::cl::desc(
          "If the symbolic optimizer is allowed (1) to operate under the "
          "finite math assumption, (2) re-order floating point operations and "
          "(3) apply reciprocal optimizations"),
      llvm::cl::init(true)};
  Option<bool> symbolicNoUndefinedMath{
      *this, "symbolic-no-undefined-math",
      llvm::cl::desc(
          "If the symbolic optimizer is allowed to assume that there will be no"
          "undefined math operations such as divisions by 0"),
      llvm::cl::init(true)};
  Option<bool> fuseJitKernels{
      *this, "fuse-jit-kernels",
      llvm::cl::desc(
          "Fusing kernels is on by default but will do nothing if useJit is "
          "off. If use jit is on then we will fuse incoming jit kernels."),
      llvm::cl::init(true)};
  Option<bool> enableProfileScopeTracing{
      *this, "enable-profile-scope-tracing",
      llvm::cl::desc("Synthesize profiler trace ranges from contiguous runs"
                     "of ops carrying the same profile scope."),
      llvm::cl::init(false)};
  Option<bool> profileScopeTracingTraceUnlabelled{
      *this, "profile-scope-tracing-trace-unlabelled",
      llvm::cl::desc("Emit ranges for regions carrying no profile scope."
                     " Useful for visualizing compiler output."),
      llvm::cl::init(false)};
  // In-process `gc-pipeline-pre-jit`, for callers that want only the emitted
  // Mojo (see `ir-output-dir`).
  Option<bool> preJit{
      *this, "pre-jit",
      llvm::cl::desc("Stop after emit-mojo instead of JIT-compiling the "
                     "kernels."),
      llvm::cl::init(false)};
  Option<bool> fuseMutableLoadStore{
      *this, "fuse-mutable-load-store",
      llvm::cl::desc("If set to false will disable the creation of in-place "
                     "kernels, instead lowering to explicit copies into and "
                     "out of mo.buffer values. Primarily used for testing."),
      llvm::cl::init(true)};
  Option<bool> usePatterns{
      *this, "use-patterns",
      llvm::cl::desc("Lower to hand-fused kernels using pattern matching."),
      llvm::cl::init(true)};
  Option<std::string> cacheDir{
      *this, "cache-dir",
      llvm::cl::desc("Path used by the CAS subsystem to establish its backing "
                     "store. If non-empty then models will be rewritten to "
                     "outline dense_attributes to the given CAS instance.")};
  ListOption<std::string> kernelLibraryPaths{
      *this, "kernel-library-paths",
      llvm::cl::desc(
          "Paths to mojo packages containing mojo kernel definitions used to "
          "replace MO ops.")};
  Option<bool> kernelDeclsPrePopulated{
      *this, "kernel-decls-pre-populated",
      llvm::cl::desc(
          "If true, the input module has already been seeded with "
          "`mogg.kernel.decl` / `mogg.primitive.decl` ops and a "
          "`mogg.opaque_type_mapping` attribute (typically copied from "
          "another module via `MOGG::copyKernelDecls`). Skips the "
          "`mogg-import-packages` pass entirely. `mogg-rewrite-opaque-types` "
          "still runs and consumes the prebuilt mapping to rewrite any "
          "`!mo.opaque<\"simple\">` references on the input ops to their "
          "fully-qualified form."),
      llvm::cl::init(false)};
  Option<bool> useShapeInference{
      *this, "mo-shape-inference",
      llvm::cl::desc("Run shape inference on the input before other "
                     "optimizations and lowering."),
      llvm::cl::init(true)};
  Option<bool> performTrivialKernelOpt{
      *this, "trivial-kernel-opt",
      llvm::cl::desc(
          "If true we will promote trivial kernels to synchronous kernels"
          "which then can be soft-fused together. This is the default but"
          "is available as a flag for tests."),
      llvm::cl::init(true)};
  Option<std::string> saveTempPrefix{
      *this, "save-temp-prefix",
      llvm::cl::desc(
          "The prefix to use when saving temporary mogg files. Setting this"
          "option to any value will toggle on mogg.saveTemps which will save"
          "assembly/IR to files this prefix."),
      llvm::cl::init("")};
  Option<bool> createMOGGReproducers{
      *this, "create-mogg-reproducers",
      llvm::cl::desc(
          "Prints isolated reproducers of each compiled mogg kernel in"
          "a newly created mo graph."),
      llvm::cl::init(false)};
  Option<bool> constantFold{
      *this, "constant-fold",
      llvm::cl::desc("Evaluate parts of the program that can be evaluated at "
                     "compile time."),
      llvm::cl::init(true)};
  Option<std::string> defaultDeviceLabel{
      *this, "default-device-label",
      llvm::cl::desc(
          "If non-empty, assign as many ops as possible to the device "
          "with the given label (and device id zero)."),
      llvm::cl::init("cpu")};
  Option<size_t> defaultDeviceId{
      *this, "default-device-id",
      llvm::cl::desc(
          "If default-device-label is specified, this field specifies the"
          "default device id to target. Defaults to zero."),
      llvm::cl::init(0)};
  Option<bool> enableCompiledRegionsFusion{
      *this, "enable-compiled-regions-fusion",
      llvm::cl::desc("If false, will disable compiled region fusion"),
      llvm::cl::init(true)};
  Option<bool> enablePostABIRegionFusion{
      *this, "enable-post-abi-regions-fusion",
      llvm::cl::desc("If false, will disable post-abi compiled region fusion"),
      llvm::cl::init(false)};
  Option<bool> debugPrintAllTensors{
      *this, "debug-print-all-tensors",
      llvm::cl::desc("If enabled, print all intermediate tensors. The usual "
                     "'result-output-style', 'print-precision' and "
                     "'binary-output-dir' flags may be used to control the "
                     "printed format and saved location."),
      llvm::cl::init(false)};
  Option<bool> nanCheck{
      *this, "nan-check",
      llvm::cl::desc("Insert NaN/Inf checks after each compiled region's "
                     "floating-point outputs."),
      llvm::cl::init(false)};
  Option<size_t> nanCheckStride{
      *this, "nan-check-stride",
      llvm::cl::desc("Sample one of every N float kernel outputs when "
                     "`nan-check` is enabled. Default 20 keeps NaN/Inf "
                     "checking usable on large models. Set to 1 to check "
                     "every tensor (slow on models with hundreds of "
                     "floating-point kernel outputs)."),
      llvm::cl::init(20)};
  ListOption<std::string> dimNames{
      *this, "dim-names",
      llvm::cl::desc("Named dimensions for each input of the MO graph")};
  Option<size_t> softFusionNonHostKernelLimit{
      *this, "soft-fusion-non-host-kernel-limit",
      llvm::cl::desc(
          "Maximum number of non-host kernels to place in a single soft fused "
          "kernel."),
      llvm::cl::init(3)};
  Option<bool> timeMOGGCompile{
      *this, "time-mogg-compile",
      llvm::cl::desc("Turns on timing for the KGEN passes invoked by mogg"),
      llvm::cl::init(false)};
  Option<bool> patternFusionMHAEnable{
      *this, "mha-enable",
      llvm::cl::desc("Enables multi-head attention (mha) pattern fusion."),
      llvm::cl::init(true)};
  Option<bool> patternFusionMHAAllowUnsafeBCasts{
      *this, "mha-allow-unsafe-bcast",
      llvm::cl::desc("If the mha pattern is enabled, allows fusing potentially "
                     "unsafe bcasts with a runtime guard."),
      llvm::cl::init(true)};
  Option<bool> patternFusionMHAAllowInternalReshape{
      *this, "mha-allow-internal-reshape",
      llvm::cl::desc("If the mha pattern is enabled, allows fusing potentially "
                     "unsafe bcasts with a runtime guard."),
      llvm::cl::init(true)};
  Option<bool> forceSync{*this, "force-sync",
                         llvm::cl::desc("Make all kernels synchronous."),
                         llvm::cl::init(false)};
  Option<int64_t> smallTensorLimit{
      *this, "small-tensor-limit",
      llvm::cl::desc(
          "Maximum number of elements a tensor may have to be considered "
          "'small' by the graph compiler. Ops which take and return only "
          "'small' tensors are themselves considered 'small', and need not be "
          "placed on the overall graph's default device."),
      llvm::cl::init(1024)};
  Option<std::string> defaultDeviceTargetTriple{
      *this, "default-device-target-triple",
      llvm::cl::desc("llvm::Triple representing the compilation target for the "
                     "defaultDevice."),
      llvm::cl::init("")};
  Option<std::string> defaultDeviceTargetArch{
      *this, "default-device-target-arch",
      llvm::cl::desc("String representing the architecture of the "
                     "defaultDevice. For example, for NVIDIA GPUs this is the "
                     "compute capability (i.e. sm_X)"),
      llvm::cl::init("")};
  Option<bool> debugKGENStubs{
      *this, "debug-kgen-stubs",
      llvm::cl::desc(
          "Create debuginfo for stubs so they show up in the debugger."),
      llvm::cl::init(false)};
  Option<bool> compileForHostExecution{
      *this, "compile-for-host-execution",
      llvm::cl::desc(
          "Compile kernels for the machine running this compilation, ignoring "
          "any virtual CPU target override. Set by sub-compilations whose "
          "artifacts execute in the compiling process (constant folding)."),
      llvm::cl::init(false)};
  Option<bool> sanityCheckShapeFuncs{
      *this, "sanity-check-shape-funcs",
      llvm::cl::desc(
          "Insert runtime checks which compare C++ and Mojo shape functions."),
      llvm::cl::init(false)};
  Option<bool> enableComputeLambdas{
      *this, "enable-compute-lambdas",
      llvm::cl::desc("If true, the Graph Compiler will generate compute "
                     "lambdas when possible"),
      llvm::cl::init(true)};
  // In-process, set via the max-debug.mlir-timing-out-dir config key since
  // torch.compile lowering can't pass cl options to MLIR utilities.
  Option<std::string> mlirTimingOutDir{
      *this,
      "mlir-timing-out-dir",
      llvm::cl::desc(
          "Folder in which MLIR pass timing in JSON files will be saved."),
  };
  Option<bool> dumpMojoText{
      *this, "dump-mojo-text",
      llvm::cl::desc("Dump generated mojo code of the kernels as text"),
      llvm::cl::init(false)};
  Option<bool> enableOpTraceComments{
      *this, "enable-op-trace-comments",
      llvm::cl::desc("Flag to enable adding comments in the Mojo code "
                     "detailing which operation is being emitted."),
      llvm::cl::init(false)};
  Option<bool> moggDebugDumpFusedTensorSpecs{
      *this, "mogg-debug-dump-fused-tensor-specs",
      llvm::cl::desc("Flag to enable adding prints of the fused tensor specs "
                     "in the lambdas of Mojo code "
                     "to easily debug / test kernel fusion."),
      llvm::cl::init(false)};
};

/// Registers the graph-compiler pipelines for testing tools, which must
/// establish the runtime the passes use.
void registerGCPipeline();
void registerGCPipelineThroughEmitMojo();
void registerMoPipeline();
void registerMapPipeline();
void registerMoggPipeline();
void registerMgpPipeline();
void registerJitPipeline();

/// Builds the complete graph-compiler pipeline (MO → MGP → JIT).
void buildGCPipeline(mlir::OpPassManager &pm, const GCPipelineOptions &opts);

/// `gc-pipeline` stopping after emit-mojo.
void buildGCPipelineThroughEmitMojo(mlir::OpPassManager &pm,
                                    const GCPipelineOptions &opts);

/// Registers the end-to-end shape inference pipeline for use by testing tools.
void registerShapeInferencePipeline();

// Create the end-to-end shape inference pipeline.
void buildShapeInferencePipeline(mlir::OpPassManager &pm);

// Sub-pipeline option structs expose only the options their stage consumes.

/// Options for `mo-pipeline`.
struct MoPipelineOptions : public mlir::PassPipelineOptions<MoPipelineOptions> {
  Option<bool> debugPrintAllTensors{
      *this, "debug-print-all-tensors",
      llvm::cl::desc("If enabled, print all intermediate tensors. The usual "
                     "'result-output-style', 'print-precision' and "
                     "'binary-output-dir' flags may be used to control the "
                     "printed format and saved location."),
      llvm::cl::init(false)};
  Option<bool> useShapeInference{
      *this, "mo-shape-inference",
      llvm::cl::desc("Run shape inference on the input before other "
                     "optimizations and lowering."),
      llvm::cl::init(true)};
  Option<bool> constantFold{
      *this, "constant-fold",
      llvm::cl::desc("Evaluate parts of the program that can be evaluated at "
                     "compile time."),
      llvm::cl::init(true)};
  Option<std::string> compiledFrameworkLabel{
      *this, "compiled-framework-label",
      llvm::cl::desc(
          "Framework label to compile to (used in mgp.model framework tag).")};
  Option<std::string> cacheDir{
      *this, "cache-dir",
      llvm::cl::desc("Path used by the CAS subsystem to establish its backing "
                     "store. If non-empty then models will be rewritten to "
                     "outline dense_attributes to the given CAS instance.")};
  Option<bool> symbolicFfastMath{
      *this, "symbolic-ffast-math",
      llvm::cl::desc(
          "If the symbolic optimizer is allowed (1) to operate under the "
          "finite math assumption, (2) re-order floating point operations and "
          "(3) apply reciprocal optimizations"),
      llvm::cl::init(true)};
  Option<bool> symbolicNoUndefinedMath{
      *this, "symbolic-no-undefined-math",
      llvm::cl::desc(
          "If the symbolic optimizer is allowed to assume that there will be no"
          "undefined math operations such as divisions by 0"),
      llvm::cl::init(true)};
  ListOption<std::string> kernelLibraryPaths{
      *this, "kernel-library-paths",
      llvm::cl::desc(
          "Paths to mojo packages containing mojo kernel definitions used to "
          "replace MO ops.")};
  Option<bool> kernelDeclsPrePopulated{
      *this, "kernel-decls-pre-populated",
      llvm::cl::desc(
          "If true, the input module has already been seeded with "
          "`mogg.kernel.decl` / `mogg.primitive.decl` ops and a "
          "`mogg.opaque_type_mapping` attribute (typically copied from "
          "another module via `MOGG::copyKernelDecls`). Skips the "
          "`mogg-import-packages` pass entirely. `mogg-rewrite-opaque-types` "
          "still runs and consumes the prebuilt mapping to rewrite any "
          "`!mo.opaque<\"simple\">` references on the input ops to their "
          "fully-qualified form."),
      llvm::cl::init(false)};
  Option<bool> usePatterns{
      *this, "use-patterns",
      llvm::cl::desc("Lower to hand-fused kernels using pattern matching."),
      llvm::cl::init(true)};
  Option<bool> patternFusionMHAEnable{
      *this, "mha-enable",
      llvm::cl::desc("Enables multi-head attention (mha) pattern fusion."),
      llvm::cl::init(true)};
  Option<bool> patternFusionMHAAllowUnsafeBCasts{
      *this, "mha-allow-unsafe-bcast",
      llvm::cl::desc("If the mha pattern is enabled, allows fusing potentially "
                     "unsafe bcasts with a runtime guard."),
      llvm::cl::init(true)};
  Option<bool> patternFusionMHAAllowInternalReshape{
      *this, "mha-allow-internal-reshape",
      llvm::cl::desc("If the mha pattern is enabled, allows fusing potentially "
                     "unsafe bcasts with a runtime guard."),
      llvm::cl::init(true)};
  Option<bool> sanityCheckShapeFuncs{
      *this, "sanity-check-shape-funcs",
      llvm::cl::desc(
          "Insert runtime checks which compare C++ and Mojo shape functions."),
      llvm::cl::init(false)};
  Option<std::string> defaultDeviceLabel{
      *this, "default-device-label",
      llvm::cl::desc(
          "If non-empty, assign as many ops as possible to the device "
          "with the given label (and device id zero)."),
      llvm::cl::init("cpu")};
  Option<size_t> defaultDeviceId{
      *this, "default-device-id",
      llvm::cl::desc(
          "If default-device-label is specified, this field specifies the"
          "default device id to target. Defaults to zero."),
      llvm::cl::init(0)};
  Option<std::string> defaultDeviceTargetTriple{
      *this, "default-device-target-triple",
      llvm::cl::desc("llvm::Triple representing the compilation target for the "
                     "defaultDevice."),
      llvm::cl::init("")};
  Option<std::string> defaultDeviceTargetArch{
      *this, "default-device-target-arch",
      llvm::cl::desc("String representing the architecture of the "
                     "defaultDevice. For example, for NVIDIA GPUs this is the "
                     "compute capability (i.e. sm_X)"),
      llvm::cl::init("")};
  Option<bool> fuseJitKernels{
      *this, "fuse-jit-kernels",
      llvm::cl::desc(
          "Fusing kernels is on by default but will do nothing if useJit is "
          "off. If use jit is on then we will fuse incoming jit kernels."),
      llvm::cl::init(true)};
};

/// Options for `map-pipeline`.
struct MapPipelineOptions
    : public mlir::PassPipelineOptions<MapPipelineOptions> {
  Option<bool> fuseMutableLoadStore{
      *this, "fuse-mutable-load-store",
      llvm::cl::desc("If set to false will disable the creation of in-place "
                     "kernels, instead lowering to explicit copies into and "
                     "out of mo.buffer values. Primarily used for testing."),
      llvm::cl::init(true)};
};

/// Options for `mogg-pipeline`.
struct MoggPipelineOptions
    : public mlir::PassPipelineOptions<MoggPipelineOptions> {};

/// Options for `mgp-pipeline`.
struct MgpPipelineOptions
    : public mlir::PassPipelineOptions<MgpPipelineOptions> {
  ListOption<std::string> kernelLibraryPaths{
      *this, "kernel-library-paths",
      llvm::cl::desc(
          "Paths to mojo packages containing mojo kernel definitions.")};
  Option<size_t> minimumCpuAlignment{
      *this, "minimum-cpu-alignment",
      llvm::cl::desc("Alignment for all CPU-allocated buffers."),
      llvm::cl::init(kPreferredMemoryAlignment)};
  Option<bool> enableOpTraceComments{
      *this, "enable-op-trace-comments",
      llvm::cl::desc("Flag to enable adding comments in the Mojo code "
                     "detailing which operation is being emitted."),
      llvm::cl::init(false)};
  Option<bool> enableComputeLambdas{
      *this, "enable-compute-lambdas",
      llvm::cl::desc("If true, the Graph Compiler will generate compute "
                     "lambdas when possible"),
      llvm::cl::init(true)};
  Option<bool> moggDebugDumpFusedTensorSpecs{
      *this, "mogg-debug-dump-fused-tensor-specs",
      llvm::cl::desc("Flag to enable adding prints of the fused tensor specs "
                     "in the lambdas of Mojo code "
                     "to easily debug / test kernel fusion."),
      llvm::cl::init(false)};
  Option<bool> dumpMojoText{
      *this, "dump-mojo-text",
      llvm::cl::desc("Dump generated mojo code of the kernels as text"),
      llvm::cl::init(false)};
  Option<bool> enableProfileScopeTracing{
      *this, "enable-profile-scope-tracing",
      llvm::cl::desc("Bracket contiguous profile scope runs of "
                     "each compiled region's body in profiler trace ranges."),
      llvm::cl::init(false)};
  Option<bool> profileScopeTracingTraceUnlabelled{
      *this, "profile-scope-tracing-trace-unlabelled",
      llvm::cl::desc("Emit the whole-region range for regions carrying no "
                     "profile scope label at all."),
      llvm::cl::init(false)};
};

/// Builds individual sub-pipelines.
void buildMoPipeline(mlir::OpPassManager &pm, const MoPipelineOptions &opts);
/// Passes between `mo-to-map` and `mogg-to-mgp`.
void buildMapPipeline(mlir::OpPassManager &pm, const MapPipelineOptions &opts);
void buildMoggPipeline(mlir::OpPassManager &pm,
                       const MoggPipelineOptions &opts);
void buildMgpPipeline(mlir::OpPassManager &pm, const MgpPipelineOptions &opts);
/// Builds the passes that run after JitCompileKernels.
void buildJitPipeline(mlir::OpPassManager &pm);

/// Runs the full gc-pipeline on `moduleOp`. With `dumpIntermediates`, writes
/// each stage's IR at its boundary. MEF conversion is left to the caller.
LogicalResult runGCPipeline(ModuleOp moduleOp, const GCPipelineOptions &opts,
                            bool dumpIntermediates = false);

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_PASSES_H
