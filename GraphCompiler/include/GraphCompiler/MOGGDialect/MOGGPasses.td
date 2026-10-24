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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD

include "mlir/Pass/PassBase.td"

def JitCompileKernels: Pass<"jit", "ModuleOp"> {
  let summary = "Replace kernel op with calls into compiled kernel";
  let description = [{
    Compiles kernel ops with the KGEN pipeline into resource blobs and replaces
    them with calls into the compiled code, inserting an op that maps kernel
    names to blobs.
  }];
  let dependentDialects = [
    "M::MDialect",
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::index::IndexDialect",
  ];

  let options = [
    Option<"saveTempPrefix", "save-temp-prefix", "std::string", /*default=*/"",
            "The prefix to use when saving temporary files. Setting this "
            "option toggles on saveTemps.">,
    Option<"dumpStub", "dump-stub", "bool", /*default=*/"false",
            "Prints the generated KGEN stub for use in tests and debugging">,
    Option<"createMOGGReproducers", "create-mogg-reproducers", "bool", /*default=*/"false",
            "Creates isolated reproducers of each kernel in a new mo graph.">,
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">,
    Option<"timeMOGGCompile", "time-mogg-compile", "bool",
      /*default=*/"false",
     "Turns on timing for the KGEN passes invoked by mogg">,
    Option<"dryRun", "dry-run", "bool",
      /*default=*/"false",
     "Run the pass but don't actually compile.">,
    Option<"debugKGENStubs", "debug-kgen-stubs", "bool",
      /*default=*/"false",
      "Create debuginfo for kgen stubs such that the debugger will give linenumbers for the stubs">,
    Option<"defaultDeviceTargetArch", "default-device-target-arch", "std::string",
      /*default=*/"",
      "Together with default-device-target-triple, overrides the accelerator "
      "arch (e.g. sm_100a) instead of probing the host GPU. Enables "
      "cross-compilation for a target GPU on a machine that lacks it (or has no "
      "GPU at all).">,
    Option<"defaultDeviceTargetTriple", "default-device-target-triple", "std::string",
      /*default=*/"",
      "Target triple (e.g. nvptx64-nvidia-cuda) identifying the accelerator "
      "vendor for default-device-target-arch. The override applies only when "
      "this triple names a GPU target.">,
    Option<"compileForHostExecution", "compile-for-host-execution", "bool",
      /*default=*/"false",
      "Compile kernels for this machine, ignoring any virtual CPU target "
      "override. For artifacts that execute in the compiling process "
      "(constant folding).">
  ];
}

def PatternFusion: Pass<"mo-pattern-fusion", "MO::GraphOp"> {
  let summary = "Fuse kernels based on patterns.";
  let description = [{
    Replaces op sequences with a call to a hand-fused kernel, or with more
    specific ops (e.g. concat with axis 0).
  }];
  let dependentDialects = ["M::MOGG::MOGGDialect"];
  let options = [
    Option<"enableMHAPattern", "mha-enable", "bool", /*default=*/"true",
            "Enable multi-head attention pattern.">,
    Option<"debugMHAPattern", "mha-debug", "bool", /*default=*/"false",
            "Enable multi-head attention pattern matching logs. Useful for debugging pattern matching failures.">,
    Option<"mhaAllowUnsafeBCast", "mha-allow-unsafe-bcast", "bool", /*default=*/"true",
            "Allow potentially unsafe broadcasts to be pattern matched for MHA patterns.">,
    Option<"mhaAllowInternalReshape", "mha-allow-internal-reshape", "bool", /*default=*/"true",
            "Allow potentially unsafe broadcasts to be pattern matched for MHA patterns.">,
    Option<"enableMegaFFNPattern", "mega-ffn-enable", "bool", /*default=*/"true",
            "Enable the NVFP4 MegaFFN fusion (grouped_matmul_swiglu_nvfp4 + "
            "grouped_matmul_block_scaled -> mega_ffn_nvfp4). Only fires on SM100 "
            "targets; the kernel is MAX-only, so this is disabled in OSS builds.">,
    Option<"targetDevice", "target-device",  "std::string", "\"\"",
           "If the graph operation does not already have a device specified, "
           "use this device for pattern fusion passes that depend on the target device.">,
    Option<"targetDeviceId", "target-device-id",  "size_t", "0",
           "If target-device is provided, this is the default-id to target">
  ];
}

def HashModuleContents: Pass<"mogg-hash-module-contents", "mlir::ModuleOp"> {
  let summary = "Compute and attach hash of module contents";
  let description = [{
    Attaches a hash of the module contents as an attribute, for caching.
  }];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
  ];
}

def NormalizeAllNames: Pass<"mogg-normalize-all-names", "mlir::ModuleOp"> {
  let summary = "Normalize all names in the module";
  let description = [{
    Renames symbols to __S0, __S1, ... and parameters to __P0, __P1, ... so
    caching and hashing ignore naming variations.
  }];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
  ];
}

def LowerCompiledRoutines: Pass<"mogg-lower-compiled-routines", "ModuleOp"> {
  let summary = "Convert MOGG compile ops into MGP versions";
  let description = [{
    Converts raw MOGG compiled kernels into the MGP format used by other
    compiled code.
  }];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def SequenceFusion: Pass<"mogg-sequence-fusion", "MGP::ModelOp"> {
  let summary = "Fuse sequences of AsyncRoutineOp";
  let description = [{}];

  let options = [
    Option<"enableTransitiveFusion", "enable-transitive-fusion", "bool", "false",
      "Flag to allow control of fusion pass">
  ];
  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect"];
}

def AddProfileScopeTracing: Pass<"mogg-add-profile-scope-tracing", "MOGG::FuncOp"> {
  let summary = "Bracket Graph profile scope ranges inside kernel stub bodies";
  let description = [{
    Synthesizes profiler trace ranges (`mogg.profile_scope.begin` /
    `mogg.profile_scope.end`) from contiguous runs of ops that share the same
    `Graph.profile_scope(...)` layer set, plus coarser ancestor ranges over
    adjacent runs nested under a common outer scope.

    Only MGP-ABI stubs are bracketed. Runs before `mogg-add-try-except-blocks`
    because begin/end can raise and must land inside its try/except.
  }];

  let options = [
    Option<"profileScopeTracingTraceUnlabelled", "profile-scope-tracing-trace-unlabelled", "bool", /*default=*/"false",
      "Emit ranges for regions carrying no profile scope. Useful for visualizing compiler output.">
  ];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def WrapFuncsWithTrace: Pass<"mogg-wrap-funcs-with-trace", "MOGG::FuncOp"> {
  let summary = "Wraps a MOGG ABI kernel stub's body in a trace range";
  let description = [{
    `mogg.with_trace` emits the per-kernel `Trace` range. MGP-ABI stubs get
    theirs from `mogg-add-profile-scope-tracing`. Runs before
    `mogg-add-try-except-blocks`, which moves this op into its try.
  }];

  let dependentDialects = ["MOGG::MOGGDialect"];
}

def AddTryExceptBlocks: Pass<"mogg-add-try-except-blocks", "MOGG::FuncOp"> {
  let summary = "Adds a try-except block to raising kernels";
  let description = [{
    Runs after sequence fusion and before Mojo codegen, so emitted Mojo for
    raising kernels handles errors.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def OutlineAsyncRoutines: Pass<"mogg-outline-async-routines", "ModuleOp"> {
  let summary = "Outline kernels into stubs";
  let description = [{
    Outlines AsyncRoutineOp into FuncOp so identical stubs can
    be merged. Runs after all optimization passes, just before Mojo codegen.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def OutlineRoutines
    : Pass<"mogg-outline-routines", "mlir::ModuleOp"> {
  let summary = "Outline each `mogg.routine` body into a `mogg.func`.";
  let description = [{
    Moves each `mogg.routine` body into a module-level `mogg.func` with
    the `MAP` ABI (so `EmitMojo` can branch on it), leaving a single
    `mogg.call` that forwards the block arguments. The `mogg.routine` is
    kept for its `!mo.tensor` interface, which `MOGGToMGP` lowers.
  }];

  let dependentDialects = [
    "M::MO::MODialect",
    "M::MOGG::MOGGDialect",
  ];
}

def LegalizeRowwiseKernels
    : Pass<"mogg-legalize-rowwise-kernels", "ModuleOp"> {
  let summary = "Fill in what each rowwise kernel needs to print as Mojo.";
  let description = [{
    Runs `mogg-rowwise-select-primary-input`, `mogg-rowwise-bind-monoids`,
    `mogg-rowwise-group-row-stores` and `mogg-rowwise-set-launch-params`, in
    that order, on every `mogg.call.rowwise`.
  }];

  let options = [
    Option<"expensiveCostThreshold", "expensive-cost-threshold", "unsigned",
           /*default=*/"16",
           "Per-element cost at which a kernel is computationally expensive. "
           "An op that divides or calls a transcendental costs 16, any other "
           "op 1.">,
  ];

  let dependentDialects = ["MOGG::MOGGDialect", "M::MO::MODialect"];
}

def RowwiseSelectPrimaryInput
    : Pass<"mogg-rowwise-select-primary-input", "ModuleOp"> {
  let summary = "Pick the input each rowwise kernel reads element by element. "
                "For testing only.";
  let description = [{
    Sets `primary` on each `mogg.call.rowwise` to the capture its first fold
    loads at the element index, and turns those loads, in every fold, into
    `mogg.rowwise.primary`. Also sets `dtype` (the primary input's element
    type) and `is_cached` (whether more than one fold reads it).
  }];

  let dependentDialects = ["MOGG::MOGGDialect"];
}

def RowwiseBindMonoids: Pass<"mogg-rowwise-bind-monoids", "ModuleOp"> {
  let summary = "Bind each fold's monoid. For testing only.";
  let description = [{
    Sets each fold's `params` (dtype parameters bound from its result types)
    and `fields` (its monoid's output field names), checks fixed field
    dtypes, and sets `accum`, the widest contribution type, on the kernel.
  }];

  let dependentDialects = ["MOGG::MOGGDialect"];
}

def RowwiseGroupRowStores: Pass<"mogg-rowwise-group-row-stores", "ModuleOp"> {
  let summary = "Move per-row stores into one emit. For testing only.";
  let description = [{
    Moves the stores at the row index of each `mogg.call.rowwise` into one
    `mogg.rowwise.emit` at the end of its body.
  }];

  let dependentDialects = ["MOGG::MOGGDialect"];
}

def RowwiseSetLaunchParams
    : Pass<"mogg-rowwise-set-launch-params", "ModuleOp"> {
  let summary = "Set each rowwise kernel's launch parameters. For testing "
                "only.";
  let description = [{
    Sets the launch parameters of each `mogg.call.rowwise`, and `cols` when
    the reduced axis is static. Runs after `mogg-rowwise-group-row-stores`: a
    body is reduce-shaped when all its stores are in the emit. Only
    reduce-shaped bodies are supported.
  }];

  let options = [
    Option<"expensiveCostThreshold", "expensive-cost-threshold", "unsigned",
           /*default=*/"16",
           "Per-element cost at which a kernel is computationally expensive. "
           "An op that divides or calls a transcendental costs 16, any other "
           "op 1.">,
  ];

  let dependentDialects = ["MOGG::MOGGDialect", "M::MO::MODialect"];
}

def AnonymizeDimNames: Pass<"mogg-anonymize-dim-names", "ModuleOp"> {
  let summary = "Drop symbolic dim names from every kernel stub's signature.";
  let description = [{
    Turns every `!mogg.dim<name>` in a `mogg.func` signature into
    `!mogg.dim`, so `MergeIdenticalFuncs` can merge stubs that differ
    only in dim names. Runs inside `EmitMojo` just before that merge.
  }];

  let dependentDialects = ["MOGG::MOGGDialect"];
}


def MergeIdenticalFuncs: Pass<"mogg-merge-identical-funcs", "ModuleOp"> {
  let summary = "Merge kernel stubs that are identical";
  let description = [{
    CSE for MOGG::FuncOp, reducing IR size and emitted Mojo.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];

  let statistics = [
    Statistic<"numMergedStubs", "# of stubs that were merged", "Number of stubs that were merged">
  ];
}

def CanonicalizeFuncsForMerge:
    Pass<"mogg-canonicalize-funcs-for-merge", "ModuleOp"> {
  let summary = "Erase what emission ignores inside kernel stubs so copies merge";
  let description = [{
    Inside each `mogg.func` body, rewrites what emission does not read, so
    copies of a kernel that differ only there merge in `MergeIdenticalFuncs`:

    - every `#M.device_ref<label, id>` (attributes and `!mgp.tensor` types) to
      id 0, since emission reads only the label (e.g. an EP-sharded MoE's
      per-device copies). See GEX-4239.
    - every dynamic dim of a `#mosh.ape` op attribute to `dyn`, since emission
      reads only the rank and static dims; runtime dims come from operands
      (e.g. per-layer copies that differ only in a symbolic dim name).

    Call sites keep real ids and names.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect"];
}

def LiftBufferCacheSlots: Pass<"mogg-lift-buffer-cache-slots", "ModuleOp"> {
  let summary = "Lift inline `index.constant` slot operands of mgp.buffer.get_cached to kernel args";
  let description = [{
    In each MGP-ABI `mogg.func`, promotes `index.constant` ops used
    only as `mgp.buffer.get_cached` slots to trailing kernel arguments,
    materialized at each `mogg.compiled.call` call site. Kernels differing
    only in slot indices can then merge. See GEX-3685.
  }];

  let dependentDialects = ["MGP::MGPDialect", "MOGG::MOGGDialect", "mlir::index::IndexDialect"];
}

def LookupConstantResources: Pass<"mogg-lookup-constant-resources", "MOGG::FuncOp"> {
  let summary = "Replace mgp.buffer.constant with mgp.resource.constant";
  let description = [{
    Reads constants from resources attached to the parent `mogg.func`,
    so kernels differing only in constant values still merge.
  }];

  let dependentDialects = ["MGP::MGPDialect"];
}

def EmitMojo: Pass<"mogg-emit-mojo", "ModuleOp"> {
  let summary = "Emit Mojo text";
  let description = [{
    Emits Mojo from kernel stubs and `mogg.routine.async`, held
    until JIT compilation.
  }];

  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions to be loaded.">,
    Option<"enableOpTraceComments", "enable-op-trace-comments", "bool", "false", "Flag to enable adding comments in the Mojo code detailing which operation is being emitted.">,
    Option<"enableComputeLambdas", "enable-compute-lambdas", "bool", "true",
      "Enable compute lambdas for the fused I/O kernels">,
    Option<"moggDebugDumpFusedTensorSpecs", "mogg-debug-dump-fused-tensor-specs", "bool", "false",
      "Flag to add prints for the fused tensor specs in the lambdas">
  ];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect",
                          "M::KGEN::KGENDialect",
                          "M::KGEN::LIT::LITDialect",
                          "M::KGEN::POP::POPDialect",
                          "M::HLCF::HLCFDialect",
                          "M::DebugInfo::DebugInfoDialect",
                          "mlir::NVVM::NVVMDialect",
                          "mlir::ROCDL::ROCDLDialect",
                          "mlir::index::IndexDialect",
  ];
}

def DumpMojo: Pass<"mogg-dump-mojo", "ModuleOp"> {
  let summary = "Dump Mojo text";
  let description = [{
    Dumps the emitted Mojo to stderr.
  }];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect",
                          "M::KGEN::KGENDialect",
                          "M::KGEN::LIT::LITDialect",
                          "M::KGEN::POP::POPDialect",
                          "M::HLCF::HLCFDialect",
                          "M::DebugInfo::DebugInfoDialect",
                          "mlir::NVVM::NVVMDialect",
                          "mlir::ROCDL::ROCDLDialect",
                          "mlir::index::IndexDialect",
    ];
}

def LoadMojo: Pass<"mogg-load-mojo", "ModuleOp"> {
  let summary = "Load Mojo text";
  let description = [{
    Replaces the module's Mojo with the given file, for editing dumped Mojo
    before running -jit-pipeline.

    Example:
    mo-opt program.mlir -gc-pipeline -o /dev/null -mlir-print-ir-after=mogg-emit-mojo &> program.mgp.mlir
    mo-opt program.mgp.mlir -mogg-dump-mojo -o /dev/null &> program.mojo
    # Modify Mojo here
    mo-opt program.mgp.mlir -mogg-load-mojo=path=program.mojo -jit-pipeline
  }];

  let options = [
    Option<"path", "path", "std::string", "\"\"",
      "The label for resulting MGP graph. Default is mof.">
  ];

  let dependentDialects = ["MEF::MEFDialect",
                          "MGP::MGPDialect",
                          "GRT::GRTDialect",
                          "MOGG::MOGGDialect",
                          "M::KGEN::KGENDialect",
                          "M::KGEN::LIT::LITDialect",
                          "M::KGEN::POP::POPDialect",
                          "M::HLCF::HLCFDialect",
                          "M::DebugInfo::DebugInfoDialect",
                          "mlir::NVVM::NVVMDialect",
                          "mlir::ROCDL::ROCDLDialect",
                          "mlir::index::IndexDialect",
  ];
}

def ImportPackages : Pass<"mogg-import-packages", "ModuleOp"> {
  let summary = "Import Mojo kernel packages and emit mogg.kernel.decl ops";
  let description = [{
    Emits one mogg.kernel.decl per registered kernel, recording its signatures,
    so later passes read them from the IR instead of loading packages. Runs
    early in the pipeline.

    - mojo-pkg-paths: compiled .mojoc files.
    - lit-mlir-paths: kgen-translate -import-mojo output; skips built-in
      packages, for self-contained lit tests.
  }];
  let options = [
    ListOption<"mojoPkgPaths", "mojo-pkg-paths", "std::string",
      "Paths to compiled .mojoc files containing kernel definitions.">,
    ListOption<"litMlirPaths", "lit-mlir-paths", "std::string",
      "Paths to MLIR files (kgen-translate -import-mojo output) for testing. "
      "Built-in packages are not loaded when this option is used alone.">,
    Option<"append", "append", "bool", /*default=*/"false",
      "Append mode: seed the registration/primitive deduplication sets from "
      "decl ops already present in the module, and skip loading built-in "
      "packages. Callers pass only the delta paths not yet loaded; existing "
      "decls in the module are preserved.">,
  ];
  let dependentDialects = [
    "M::MOGG::MOGGDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::index::IndexDialect",
  ];
}

def RewriteOpaqueTypes : Pass<"mogg-rewrite-opaque-types", "ModuleOp"> {
  let summary = "Rewrite !mo.opaque<\"simple\"> to !mo.opaque<\"fully.qualified\">";
  let description = [{
    Rewrites each `!mo.opaque<"simple_name">` to its fully-qualified name using
    the module's `mogg.opaque_type_mapping` (written by `mogg-import-packages`).
    Unmapped types are unchanged. Kept separate so callers that loaded packages
    elsewhere (e.g. `MAXG_addKernelPackage`) can copy the mapping and run only
    this step; see the `kernel-decls-pre-populated` pipeline option.
  }];
}

def StripKernelDecls : Pass<"mogg-strip-kernel-decls", "ModuleOp"> {
  let summary = "Remove mogg.kernel.decl and mogg.primitive.decl ops from a module";
  let description = [{
    Run after the last pass that reads the kernel registry; the decls are not
    part of the final output.
  }];
}

def VerifyKernels: Pass<"mogg-verify-kernels", "ModuleOp"> {
  let summary = "Verifies that kernels are well constructed";
  let description = [{
    Emits user-facing diagnostics for malformed `mogg.kernel.decl` signatures
    (missing `execute`, mismatched shape/execute inputs, ...), located at the
    offending Mojo argument or entry point when the decl records it.
  }];
}

def DumpKernels: Pass<"mogg-dump-kernels", "ModuleOp"> {
  let summary = "Prints the internal representation of kernels";
  let description = [{
    For debugging. Works on both `LIT` and `KGEN` kernels.
  }];
}

def ConformCallOperands
    : Pass<"mogg-conform-call-operands", "mlir::ModuleOp"> {
  let summary = "Conform `mogg` call operands to the callee's signature.";
  let description = [{
    Wraps scalar arguments in `mogg.scalar.create`, packs variadic runs into
    `mogg.tuple.create`, and wraps trait-bound arguments in
    `mogg.trait.create`. Kept out of `MAPToMOGG` so fusion IR stays simpler and
    each `::emitMojo` is a local, 1:1 lowering.
  }];

  let dependentDialects = [
    "M::MOGG::MOGGDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGPASSES_TD
