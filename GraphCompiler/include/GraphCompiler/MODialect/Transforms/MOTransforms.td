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

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def AddFallbackShapeFunctions :
    Pass<"mo-add-fallback-shape-funcs", "GraphOp"> {
  let summary = "Adds fallback shape functions to unmaterialized mo ops";
  let description = [{
    Gives every MO op parameterized outputs, materializing a fallback shape
    function for ops that lack them.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::index::IndexDialect",
    "M::DebugInfo::DebugInfoDialect",
    "M::HLCF::HLCFDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::MDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::ub::UBDialect",
  ];
}

def InvokeShapePackingFunctions : Pass<"mo-invoke-shape-packing-funcs", "GraphOp"> {
  let summary = "Process fallback shape functions for packing operations";

  let description = [{
    Uses a C++ re-implementation of the shape calculations.
  }];

  let options = [
    Option<"useHostSystem", "use-host-system", "bool",
      /*default=*/"true",
      "Use host target info instead of the GraphOp's M.device attribute. "
      "Disable for deterministic FileCheck output.">,

    Option<"sanityCheckShapeFuncs", "sanity-check-shape-funcs", "bool",
      /*default=*/"false",
      "Emit IR asserts that the C++ shapes match the Mojo implementation's.">,
  ];

  let dependentDialects = [
    "M::KGEN::KGENDialect"
  ];
}

def MergeDuplicateShapeMaterializations : Pass<"mo-merge-duplicate-shape-materializations", "GraphOp"> {
  let summary = "Merges duplicate shape materialization operations";
  let description = [{
    Dedupes Staticization ops of the same type on the same operands, so one
    value doesn't get multiple shape parameters.
    Example:
    ```
    mo.graph @example<N1 -> O1>(%arg0: !mo.tensor<[], bf16>, %arg1: !mo.tensor<[N1], bool>) -> !mo.tensor<[O1], bf16> {
      %0 = mo.shape_of(%arg1) : (!mo.tensor<[N1], bool>) -> !mo.tensor<[1], si32>
      %1 = mo.broadcast_to<() -> D1>(%arg0, %0) : (!mo.tensor<[], bf16>, !mo.tensor<[1], si32>) -> !mo.tensor<[D1], bf16>
      %2 = mo.rebind(%1) : !mo.tensor<[D1], bf16> -> !mo.tensor<[N1], bf16>
      %3 = mo.broadcast_to<() -> D2>(%arg0, %0) : (!mo.tensor<[], bf16>, !mo.tensor<[1], si32>) -> !mo.tensor<[D2], bf16>
      %4 = mo.rebind(%3) : !mo.tensor<[D2], bf16> -> !mo.tensor<[N1], bf16>
      %5 = mo.select(%arg1, %2, %4) : (!mo.tensor<[N1], bool>, !mo.tensor<[N1], bf16>, !mo.tensor<[N1], bf16>) -> !mo.tensor<[N1], bf16>
      mo.output<N1> %5 : !mo.tensor<[N1], bf16>
    }
    ```
    becomes
    ```
    mo.graph @example<N1 -> O1>(%arg0: !mo.tensor<[], bf16>, %arg1: !mo.tensor<[N1], bool>) -> !mo.tensor<[O1], bf16> {
      %0 = mo.shape_of(%arg1) : (!mo.tensor<[N1], bool>) -> !mo.tensor<[1], si32>
      %1 = mo.broadcast_to<() -> D1>(%arg0, %0) : (!mo.tensor<[], bf16>, !mo.tensor<[1], si32>) -> !mo.tensor<[D1], bf16>
      %2 = mo.rebind(%1) : !mo.tensor<[D1], bf16> -> !mo.tensor<[N1], bf16>
      %3 = mo.rebind(%1) : !mo.tensor<[D1], bf16> -> !mo.tensor<[N1], bf16>
      %4 = mo.select(%arg1, %2, %3) : (!mo.tensor<[N1], bool>, !mo.tensor<[N1], bf16>, !mo.tensor<[N1], bf16>) -> !mo.tensor<[N1], bf16>
      mo.output<N1> %4 : !mo.tensor<[N1], bf16>
    }
    ```
    Canonicalization and CSE then clean up further.
  }];
}

def PropagateShapes : Pass<"mo-propagate-shapes", "GraphOp"> {
  let summary = "Propagates shape parameters through the graph.";
  let description = [{
    Evaluates and simplifies shape parameter logic in the graph.
  }];
}

def SymbolicOptimization : Pass<"mo-symbolic-optimize", "GraphOp"> {
  let summary = "Optimize the graph using symbolic reasoning.";
  let description = [{
    Uses symbolic information (usually from shape inference) to move ops,
    rewrite op sequences into cheaper ones, and canonicalize ops for
    downstream optimizations.
  }];

  let options = [
    Option<"symbolicFfastMath", "symbolic-ffast-math", "bool", "true",
      "Allow assuming finite math, re-ordering floating point ops, and "
      "reciprocal optimizations">,
      Option<"symbolicNoUndefinedMath", "symbolic-no-undefined-math", "bool", "true",
      "Allow assuming no undefined math such as division by 0">
  ];
}

def InferLayouts : Pass<"mo-infer-layouts", "GraphOp"> {
  let summary = "Infer layouts of tensor values";
  let description = [{
    Infers tensor layouts from op constraints (e.g. `mo.conv` assumes `NHWC`
    input before layout transformations) and annotates the tensor types.
  }];
}

def HoistParameterExpressions : Pass<"mo-hoist-param-exprs", "GraphOp"> {
  let summary = "Move parameter expressions out of ops and into a preceding alias.";
  let description = [{
    Transforms
    ```
    %1 = mo.op(%arg: !mo.tensor<[add(N,M)], f32>) : !mo.tensor<[add(N,M)], f32>
    ```
    into
    ```
    kgen.param.declare param_expr = <add(N,M)>
    %arg_rebound = mo.rebind(%arg:!mo.tensor<[add(N,M)], f32>) :!mo.tensor<[param_expr], f32>
    %tmp = mo.op(%arg_rebound: !mo.tensor<[param_expr], f32>) : !mo.tensor<[param_expr], f32>
    %1 = %mo.rebind(tmp: !mo.tensor<(N,M), f32>): !mo.tensor<[add(N,M)], f32>
    ```
    The `rebind`s keep the block's input and output types unchanged. MOGG
    needs this to jit kernels with parameter expressions in their arguments.
  }];
}

def ConstantFold : Pass<"mo-constant-fold", "ModuleOp"> {
  let summary = "Partially evaluate constant subgraphs";
  let description = [{
    Executes constant subgraphs at compile time with the kernel
    infrastructure. Does not explicitly handle control flow.

    Example:
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si32>
      %1 = mo.constant <#M.dense_array<6, 3> : tensor<2xsi32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si32>
      %2 = mo.mul(%0, %1) : !mo.tensor<[2], si32>
      %3 = mo.sub(%2, %1) : !mo.tensor<[2], si32>
      mo.output %3 : !mo.tensor<[2], si32>
    }
    ```
    becomes
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant <#M.dense_array<0, 2> : tensor<2xsi32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si32>
      mo.output %0 : !mo.tensor<[2], si32>
    }
    ```

    Output node order may be non-deterministic.

    Runs on ModuleOps because runPipeline(MOToMOGG) cannot nest in a GraphOp,
    and to avoid repeating lowering and model setup.
  }];
  // Required by the nested MOToMOGG pipeline (MAP under advanced fusion).
  let dependentDialects = [
    "M::MDialect",
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
    "M::MGP::MGPDialect",
    "M::GRT::GRTDialect",
    "M::KGEN::KGENDialect",
    "M::KGEN::LIT::LITDialect",
    "M::KGEN::POP::POPDialect",
    "M::HLCF::HLCFDialect",
    "M::DebugInfo::DebugInfoDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::index::IndexDialect",
    "mlir::ub::UBDialect",
  ];
  let options = [
    Option<"iterations", "iterations", "int", "2", "Number of times to run constant folding back to back">,
  ];
}

def ConstantFoldSubgraphTest : Pass<"mo-constant-fold-subgraph-test", "ModuleOp"> {
  let summary = "Test constant subgraph detection";
  let description = [{
    Test-only: prints the graph, then one `subgraph` line per constant
    subgraph found:
    ```
    mo.graph @const_program() -> !mo.tensor<[2], si32> {
      %0 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si32>
      %1 = mo.constant <#M.dense_array<6, 3> : tensor<2xsi32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2], si32>
      %2 = mo.mul(%0, %1) : !mo.tensor<[2], si32>
      %3 = mo.sub(%2, %1) : !mo.tensor<[2], si32>
      mo.output %3 : !mo.tensor<[2], si32>
    }
    subgraph %0 %1 %2 %3
    ```
  }];
}

def SupportConversionsTest : Pass<"mo-support-conversions-test", "ModuleOp"> {
  let summary = "Testing only pass for testing MO Conversion support library.";
  let description = [{
    Test-only: expands hard-coded unregistered ops with the helpers in
    GraphCompiler/Conversion/MO/Support/*.
    ```
    mo.graph @const_program() -> (!mo.tensor<[2], f32>) {
      ...
      // Expanded into decomposed MO ops:
      %2 = "prelu"(%0, %1) : (!mo.tensor<[2], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[2], f32>
      mo.output %2 : !mo.tensor<[5], f32>
    }
    ```
    See SupportConversionsTest.cpp for more registered rewrite patterns.
  }];

  let dependentDialects = [
    "M::RMO::RMODialect"
  ];
}

def ValidateDevicesPass : Pass<"mo-validate-devices", "GraphOp"> {
  let summary = "Validates that each data type used in tensors in the graph "
                "can be implemented on the target device.";
  let description = [{
    Checks the graph's inputs and outputs.
  }];

  let options = [
    Option<"defaultDeviceTargetTriple", "default-device-target-triple", "std::string", "\"\"",
      "llvm::Triple of the defaultDevice's compilation target.">,
    Option<"defaultDeviceTargetArch", "default-device-target-arch", "std::string", "\"\"",
      "Architecture of the defaultDevice (e.g. sm_X on NVIDIA GPUs).">,
  ];
}

def VerifyDeviceGraphPass : Pass<"mo-verify-device-graph", "GraphOp"> {
  let summary = "Verify that a device-graphable mo.graph has kernel "
                "registrations that expose a build_graph entry point.";
  let description = [{
    In a `mo.graph` with `isDeviceGraph`, errors on any op whose
    `MojoRegistry` kernel registration lacks `build_graph`. Unregistered ops
    (e.g. `mo.constant`, `mo.output`) are skipped.
  }];
}

def DebugPrintAllTensorsPass : Pass<"mo-debug-print-all-tensors", "GraphOp"> {
  let summary = "Print all intermediate tensors as they are produced.";
  let description = [{
    Emits mo.debug.tensor.print for every mo.tensor value, inputs included.
    The graph is topologically sorted first so the chains serializing the
    prints can't form a cycle. For debugging only.
  }];
}

def NanCheckPass : Pass<"mo-nan-check-pass", "GraphOp"> {
  let summary = "Insert NaN/Inf checks after each compiled region's floating-point outputs.";
  let description = [{
    Inserts nan_check ops (lowered to MOGG kernels in place) after
    floating-point tensor outputs. Each returns a pass-through copy that
    downstream uses are rewired to, preventing DCE. For debugging only,
    enabled via the `max-debug.nan-check` config key or `--nan-check`.

    `stride` checks one of every N float kernel outputs: checking all of them
    on large models (e.g. ~920 in gemma-4-31b-it) overwhelms the
    single-threaded Mojo/KGEN compile path (GEX-3703). Use stride=1 on small
    graphs such as lit tests.
  }];
  let options = [
    ListOption<"kernelLibraryPaths", "kernel-library-paths", "std::string",
      "Paths to mojo packages containing mojo kernel definitions.">,
    Option<"stride", "stride", "uint64_t", /*default=*/"20",
      "Check one of every N float kernel outputs (1 = check all).">
  ];
}

def SplatLargeConstants :
    Pass<"mo-splat-large-constants", "GraphOp"> {
  let summary = "Will splat a large constant for development purposes";
  let description = [{
    Replaces large constants with a broadcast scalar. Does not preserve
    correctness; for development only.
  }];

  let options = [
    Option<"numElements", "num-elements", "int", "100000000",
      "Threshold number of elements in the constant's dense array">,
  ];
}

def TopologicalSort :
    Pass<"mo-topological-sort", "ModuleOp"> {
  let summary = "Topologically sort ops in a graph based on SSA and parameter dependencies.";
  let description = [{
    Handles nested regions (e.g. `mo.if`).
  }];
}

def ElideRedundantMutableOps :
    Pass<"mo-elide-redundant-mutable-ops", "GraphOp"> {
  let summary = "Eliminates redundant mo.mutable.store and mo.mutable.load ops.";
  let description = [{
    Lets the MOGG fusion passes work with in-place operations.

    A load is erased when it consumes the output chain of a store to the
    same buffer:

      e.g.

      ```
      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor0)
      %ch2, %tensor1 = mo.mutable.load(%ch1, %buffer)

      %res = mo.op(%tensor1, ...)

      %ch3 = mo.mutable.store(%ch2, %buffer, %res)

      ==>

      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor0)

      %res = mo.op(%tensor0, ...)

      %ch3 = mo.mutable.store(%ch1, %buffer, %res)
      ```

    A store is erased when a later store to the same buffer consumes its
    output chain:

      ```
      %ch1 = mo.mutable.store(%ch0, %buffer, %tensor)
      %res = mo.op(%tensor, ...)
      %ch3 = mo.mutable.store(%ch1, %buffer, %res)

      ==>

      %res = mo.op(%tensor, ...)
      %ch3 = mo.mutable.store(%ch0, %buffer, %res)
      ```
  }];
}

def FlattenElementwise : Pass<"mo-flatten-elementwise", "ModuleOp"> {
  let summary = "Flattens elementwise ops";
  let description = [{
    Flattens elementwise op shapes when that unblocks elementwise fusion.
  }];
}

def CanonicalizeChainSequences : Pass<"mo-canonicalize-chain-sequences", "GraphOp"> {
  let summary = "Optimize linear chains of side-effecting operations.";
  let description = [{
    Given two separate mo.buffers:

    ```mlir
    %ch1, %vbuf = mo.mutable.load(%ch0, %buf0)
    ...
    %ch2 = mo.mutable.store(%ch1, %buf0, %res1)
    ...
    %ch3, %vbuf2 = mo.mutable.load(%ch2, %buf1)
    ...
    %ch4 = mo.mutable.store(%ch3, %buf1, %res2)
    ```

    the shared chain makes the ops on `%buf1` needlessly wait for those on
    `%buf0`. Local transforms rewrite this to:

    ```mlir
    %ch1, %vbuf = mo.mutable.load(%ch0, %buf0)
    ...
    %ch2 = mo.mutable.store(%ch1, %buf0, %res1)
    ...
    %ch3, %vbuf2 = mo.mutable.load(%ch0, %buf1) // <- note the difference here
    ...
    %ch4 = mo.mutable.store(%ch3, %buf1, %res2)
    ```
  }];
}

def HoistConstantSubgraphsIntoParent : Pass<"mo-hoist-constant-subgraphs-into-parent", "ModuleOp"> {
  let summary = "Hoist constant and constant-derived subgraphs to the parent graph.";
  let description = [{
    ExecInvariantCodeMotion doesn't support subgraphs, so this lets constant
    ops be constant-folded or hoisted to the init block.
  }];
}

def RemoveUnusedOps : Pass<"remove-unused-ops", "GraphOp"> {
  let summary = "Remove unused (dead) MO operations from the graph";
  let description = [{
    Erases unused ops that CSE keeps for their memory effects. Chains encode
    their dependencies, so an op with an unused output chain is dead.
  }];
}

def CreateParallelFromUnbundle :
    Pass<"mo-create-parallel-from-unbundle", "GraphOp"> {
  let summary =
      "Wrap per-device consumers of an mo.tensor.unbundle in a fresh mo.parallel.";
  let description = [{
    For each `mo.tensor.unbundle` whose N results each have one user, all
    structurally identical, creates an `mo.parallel` holding one clone of the
    consumer. Its primary input is bundled from the unbundle's results;
    other per-launch-varying operands become new bundled inputs.

    The transient `bundle(unbundle(X)...)` canonicalizes to `X`, leaving the
    region adjacent to the producer for `mo-merge-adjacent-parallels`.
  }];
}

def MergeAdjacentParallels :
    Pass<"mo-merge-adjacent-parallels", "GraphOp"> {
  let summary =
      "Merge an mo.parallel that consumes another mo.parallel's bundle output.";
  let description = [{
    Peephole: when `mo.parallel` `%B` consumes bundle results of `%A`, B's
    body is spliced into A's, shared inputs are short-circuited via A's yield
    operands, and B's yields are appended to A's.

    Buffer and chain rules:

      * If both A and B have buffers + chain, the buffer operand lists must
        be structurally identical, and B's in-chain must be A's out-chain
        with no other external users.  The merged op inherits A's buffers
        and in-chain; its out-chain replaces B's out-chain.
      * If only one side has buffers + chain, the merged op inherits that
        side's buffers and chain.
      * Neither: the merged op is chainless.

    Otherwise the pair is left unmerged.
  }];
}

def MarkStableTensorsPass : Pass<"mo-mark-stable-tensors", "GraphOp"> {
  let summary = "Route operands that a kernel's build_graph declares stable "
                "through mo.stable.create.";
  let description = [{
    In a `mo.graph` carrying the `isDeviceGraph` attribute, resolves every
    kernel-backed op to its `mogg.kernel.decl` and maps the op's operands onto
    the kernel's `build_graph` signature. Each operand that lands on a
    `StableTensor` / `StableVariadicTensors` argument is rewritten to go
    through `mo.stable.create`, so the operand type carries the `stable` bit
    that kernel-stub setup and emission key off.

    Graphs without the `isDeviceGraph` attribute are no-ops.
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_TD
