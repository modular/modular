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

#ifndef GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
#define GENERICML_MGPDIALECT_MGPTRANSFORMS_TD

include "mlir/Pass/PassBase.td"
include "GraphCompiler/MGPDialect/Dialect/MGPOps.td"

//===----------------------------------------------------------------------===//
// AsyncDataflowDependenciesTestPass
//===----------------------------------------------------------------------===//

def AsyncDataflowDependenciesTestPass : Pass<"mgp-async-dataflow-dependencies-test", "MGP::ModelOp"> {
  let summary = "Prints the sameAs and impliesReady relations for all pairs of "
                "values in the input model, leaving the model unchanged. "
                "For testing only.";
}

//===----------------------------------------------------------------------===//
// SimplifyExtendLifetimesPass
//===----------------------------------------------------------------------===//

def SimplifyExtendLifetimesPass : Pass<"mgp-simplify-extend-lifetimes", "MGP::ModelOp"> {
  let summary = "Simplify extend lifetime ops";
  let description = [{
    Given:
    ```mlir
      %ch1 = mogg.routine.async(...) -> !mef.chain
      %ch2 = mgp.extend.lifetime %ch1, %t1, ... %tn
      op(%ch2)
    ```
    We can fuse these into simply:
    ```mlir
      %ch1 = mogg.routine.async(..., %t1, ... %tn) -> !mef.chain
      op(%ch1)
    ```
    Provided they are not then removed for being unused. So we must run this pass last.
  }];
}

//===----------------------------------------------------------------------===//
// SimplifyChainsPass
//===----------------------------------------------------------------------===//

def SimplifyChainsPass : Pass<"mgp-simplify-chains", "MGP::ModelOp"> {
  let summary = "Removed unnecessary chains from grt.chain.create ops.";
  let description = [{
    Given:
    ```mlir
      %a = grt.chain.create (%x, %y, %z)
    ```
    where AsyncValueDependencies::impliesReady(%x, %y), %y is dropped:
    ```mlir
      %a = grt.chain.create (%x, %z)
    ```

    If also impliesReady(%x, %z), %a is replaced by %x. Operands known to be
    isAlwaysReady are removed.
  }];

  let statistics = [
    Statistic<"numEliminatedChains", "num-eliminated-chains",
              "the number of chain operands eliminated">,
  ];
}

//===----------------------------------------------------------------------===//
// DeferAllocsPass
//===----------------------------------------------------------------------===//

def DeferAllocsPass : Pass<"mgp-defer-allocs", "MGP::ModelOp"> {
  let summary = "Defers allocations until the producing op is ready to execute.";
  let description = [{
    Ensures mgp.buffer.alloc.* operations are deferred till they are
    about to be written to. Eg:
    ```mlir
      %buf = mgp.buffer.alloc %bufChain, 24
      my_op(%opChain, %buf, ...)
      // where %buf is in an 'output' position
    ```
    becomes:
    ```
      %ch = grt.chain.create (%opChain, %bufChain)
      %buf = mgp.buffer.alloc %ch, 24
      my_op(%ch, %buf, ...)
    ```
    Assumes:
      - Ops with buffer operands have "numOutputBuffers" attributes.
  }];
}

//===----------------------------------------------------------------------===//
// ConcatInPlacePass
//===----------------------------------------------------------------------===//

def ConcatInPlacePass : Pass<"mgp-concat-in-place", "MGP::ModelOp"> {
  let summary = "Replaces `mgp.buffer.concat` with `mgp.buffer.split.borrow`.";
  let description = [{
    Applies when all writers to the concat's operand buffers can be rewritten
    to write to the split's result buffers instead.

    For example:
    ```mlir
    %in1_buf = mgp.buffer.alloc ... : !mgp.buffer<size1>
    %in1_ch = ... writer to %in1_buf ...
    %in2_bytes = ...
    %in2_buf = mgp.buffer.alloc ..., %in2_bytes
    %in2_ch = ... writer to %in2_buf ...
    %out_bytes = ...
    %out_buf = mgp.buffer.alloc ..., %out_bytes
    %ch1 = grt.chain.create (%in1_ch, %in2_ch)
    %ch2 = mgp.buffer.concat %ch1, %out_buf, %in1_buf, %in2_buf : ...
    // where these are the only uses of %in1_buf and %in2_buf
    ```
    is rewritten to:
    ```mlir
    %out_bytes = ...
    %out_buf = mgp.buffer.alloc %root_ch, %out_bytes
    %in2_spec = ...
    %in2_size = ...
    %in1_buf, %in2_buf = mgp.buffer.split %root_ch, %out_buf, %in2_size : ...
    %in1_ch = ... writer to %in1_buf ...
    %in2_ch = ... writer to %in2_buf ...
    %ch2 = grt.chain.create (%in1_ch, %in2_ch)
    ```

    Assumes the concat's argument chain is ready only once all argument
    buffers are, so uses of %ch2 are replaced with %ch1.
  }];
}

//===----------------------------------------------------------------------===//
// ExecInvariantCodeMotionPass
//===----------------------------------------------------------------------===//

def ExecInvariantCodeMotionPass : Pass<"mgp-exec-invariant-code-motion", "MGP::ModelOp"> {
  let summary = "Moves code from the 'execute' to the 'init' blocks.";
  let description = [{
    This pass moves computations which do not depend on the 'execute' block
    arguments (directly or indirectly) to the 'init' block, using slots in
    the `!mgp.context` to forward the results, subject to heuristics to:
      - avoid moving 'trivial' code, where the cost of caching and retrieving
        the result outweighs the cost of the computation itself;
      - avoid moving code whose result is awkward to forward within the
        `!mgp.context`.
      - avoid caching results which are now only needed within the 'init' block.

    Notes:
      1. Lifted buffers are cached in 'init' and looked up in 'execute'.
          (a). Buffers whose users are all lifted are not cached.
          (b). Buffer ops with no lifted user are not lifted, since that only
               adds caching without saving computation.

      2. Ops modifying lifted buffers (e.g., matmul weights pre-packing) are
         lifted too. To preserve their chain dependency:
          (a). 'init' outputs a chain accounting for the lifted chain.
          (b). 'execute' uses the input chain instead of the lifted chain.

      3. Lifted ops with non-buffer outputs are recomputed: a clone stays in
         'execute', since these types have no cache. For instance, the output
         spec of matmul weight packing is used by both the lifted packing and
         the matmul in 'execute'.

    For example:
    ```mlir
    mgp.model @example : mof init {
      %ch0 = grt.chain.create ()
      %ctx = mgp.context.create 1
      mef.output %ch0, %ctx : !mef.chain, !mgp.context

    } execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg: !mgp.tensor<?xf32>) -> (!mef.chain, !mgp.tensor<?xf32>) {
      %const_buf     : !mgp.buffer
      %const_spec    : !mgp.tensor_spec<?xf32>
      %new_const_buf : !mgp.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : (!mgp.tensor_spec<?xf32>) -> !mgp.tensor_spec<?xf32>
      %kernel_ch1 = kernel_op1(%ch0, %const_buf, %const_spec, %new_const_buf, %new_const_spec) : ... -> mef.chain

      %out_spec : !mgp.tensor_spec<?xf32>
      %out_buf  : !mgp.buffer
      %kernel_ch2 = kernel_op2(%kernel_ch1, %arg_buf, %arg_spec, %new_const_buf, %new_const_spec, %out_buf, %out_spec) : ...
      %out_tensor = mgp.tensor.create.deprecated %kernel_ch2, %out_buf, %out_spec  : !mgp.tensor<?xf32>
      mef.output %ch0, %out_tensor : !mef.chain, !mgp.tensor<?xf32>
    }
    ```
    is rewritten to:
    ```mlir
    mgp.model @example : mof init {
      %ch0 = grt.chain.create ()
      // number of slots increased, to store lifted buffers
      %ctx = mgp.context.create 2

      // moved from exec phase
      %const_buf     : !mgp.buffer
      %const_spec    : !mgp.tensor_spec<?xf32>
      %new_const_buf : !mgp.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : ...
      %kernel_ch1 = kernel_op1(%ch0, %const_buf, %const_spec, %new_const_buf, %new_const_spec) : ... -> mef.chain

      // inserted; buffer.cache may perform a deep copy, so we must wait for the
      // actions on buffers to finish first.
      %buffer_action_ch = grt.merge (%kernel_ch1) // may have more action chain in general
      %cache_ch = mgp.buffer.cache %buffer_action_ch, %ctx[1], %new_const_buf : !mgp.buffer

      // out chain changed
      %out_ch = grt.merge (%cache_ch) // may have more cache_ch in general
      mef.output %out_ch, %ctx : !mef.chain, !mgp.context

    } execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg: !mgp.tensor<?xf32>) -> (!mef.chain, !mgp.tensor<?xf32>) {
      // cloned
      %const_spec : !mgp.tensor_spec<?xf32>
      %new_const_spec = spec_update_op(%const_spec) : (!mgp.tensor_spec<?xf32>) -> !mgp.tensor_spec<?xf32>

      // inserted
      %new_const_buf = mgp.buffer.get_cached %ch0, %ctx[1] : !mgp.buffer

      // references to `%kernel_ch1` become `%ch0`
      %out_spec : !mgp.tensor_spec<?xf32>
      %out_buf  : !mgp.buffer
      %kernel_ch2 = kernel_op2(%ch0, %arg_buf, %arg_spec, %new_const_buf, %new_const_spec, %out_buf, %out_spec) : ... -> mef.chain
      %out_tensor = mgp.tensor.create.deprecated %kernel_ch2, %out_buf, %out_spec  : !mgp.tensor<?xf32>
      mef.output %ch0, %out_tensor : !mef.chain, !mgp.tensor<?xf32>
    }
    ```

    TODO(#24751): Possible improvements to this pass are tracked in this issue.
  }];
}

//===----------------------------------------------------------------------===//
// FixupOverhangsPass
//===----------------------------------------------------------------------===//

def FixupOverhangsPass : Pass<"mgp-fixup-overhangs", "MGP::ModelOp"> {
  let summary = "If necessary, inserts additional chain dependences to ensure "
                "no overhanging chains in the graph.";
  let description = [{
    This pass checks for overhanging chains (i.e., chains that have no uses in
    the graph) and inserts dependencies on the graph output to ensure that the
    output waits on all previously hanging chains.
  }];
}

//===----------------------------------------------------------------------===//
// PredicateAssertDependencies
//===----------------------------------------------------------------------===//

def PredicateAssertDependenciesPass : Pass<"mgp-predicate-asserts-deps", "MGP::ModelOp"> {
  let summary = "Predicates the execution of operations that directly or "
                "indirectly depend the result of an assert operation.";
  let description = [{
    For each block in the IR, any operation that takes in a chain and uses a
    value that contributed to at least one assert will have to depend on the
    chains of those asserts.

    Before
    ------
    %0 = some_op
    %1 = mgp.assert %0
    some_user %ch, %0

    After
    ------
    %0 = some_op
    %1 = mgp.assert %0
    %ch2 = grt.chain.create %ch, %1
    some_user %ch2, %0

    Ops that take no chain get no added dependency, so their results may still
    be computed from values whose assert fails.
  }];
}

//===----------------------------------------------------------------------===//
// VerifyModelArgumentsPass
//===----------------------------------------------------------------------===//

def VerifyModelArgumentsPass : Pass<"mgp-verify-model-arguments", "MGP::ModelOp"> {
  let summary = "Ensures that user provided model arguments match the static "
                "expectations";
  let description = [{
    Adds runtime shape checks that user-provided tensors match the model
    arguments' static expectations.

    Example:
    ```mlir
    %known = index.constant 4
    %dim = mgp.tensor_spec.get_dim[0] %arg2 : !mgp.tensor_spec<4xf32>
    %valid = index.cmp eq(%dim, %known)
    %ch = mgp.assert %valid, "Tensor type mismatch for argument 0"
    ```
  }];
  let dependentDialects = ["KGEN::POP::POPDialect"];
}

//===----------------------------------------------------------------------===//
// PrepareDevicePrimitivesPass
//===----------------------------------------------------------------------===//

def PrepareDevicePrimitivesPass : Pass<"mgp-prepare-device-primitives"> {
  let summary = "Prepares device-generic MGP primitives for execution by the "
                "GraphRT.";
  let description = [{
    This pass:
      - appends the label of each device-generic MGP primitive's 'main' device
        to its name, so GraphRT primitive dispatch supports multiple devices
        without extra indirection.
      - replaces #M.device_ref attributes with just the device id, avoiding
        reconstructing device labels at runtime that are ignored anyway.

    For example:
    ```mlir
      %buf = mgp.buffer.alloc %ch, %size: !mgp.buffer<<"gpu", 3>>
    ```
    becomes:
    ```mlir
      %buf = "mgp.buffer.alloc.gpu" (%ch, %ctx)
               { aRuntimeSlot = 1 : ui64,
                 bRawAlign = kUnknownSize,
                 cDevice = 3 : ui64 }
               : (!mef.chain, !mgp.context) ->
                 !mgp.buffer<<"gpu", 3>>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExplicitBufferLifetimesPass
//===----------------------------------------------------------------------===//

def ExplicitBufferLifetimesPass : Pass<"mgp-buffer-lifetimes", "MGP::ModelOp"> {
  let summary = "Inserts mgp.buffer.free ops based on the results of MGPBufferLifetimeAnalysis";
  let description = [{
    Inserts an mgp.buffer.free op for each mgp.alloc where the buffer is
    guaranteed to have no more accesses, if such a point exists before the end
    of the graph.

    The buffer.free ops do not free anything (the runtime refcounts buffers);
    this pass only exists to debug and validate BufferLifetimeAnalysisPass.
  }];
}

//===----------------------------------------------------------------------===//
// SimpleBufferReusePass
//===----------------------------------------------------------------------===//

def SimpleBufferReusePass : Pass<"mgp-simple-buffer-reuse", "MGP::ModelOp"> {
  let summary = "Reuses allocated buffers in simple cases";
  let description = [{
    This pass replaces a mgp.buffer.alloc op B with a
    mgp.buffer.alloc op A if the following conditions are met:
      1) all uses of A are guaranteed to execute before the first use of B
      2) A has the same runtime size SSA value as B
      3) A has the same input chain as B
      4) A has the same device as B

    For example:
    ```mlir
      %input = ...
      %inChain = ...
      %size = ...
      %buf0 = mgp.buffer.alloc %inChain, %size, ...
      %buf1 = mgp.buffer.alloc %inChain, %size, ...
      %buf2 = mgp.buffer.alloc %inChain, %size, ...

      %outChain_0 = use(%inChain, %input, %buf0)
      %outChain_1 = use(%outChain_0, %buf0, %buf1)
      %outChain_2 = use(%outChain_1, %buf1, %buf2)
    ```
    becomes:
    ```mlir
      %input = ...
      %inChain = ...
      %size = ...
      %buf0 = mgp.buffer.alloc %inChain, %size, ...
      %buf1 = mgp.buffer.alloc %inChain, %size, ...

      %outChain_0 = use(%inChain, %input, %buf0)
      %outChain_1 = use(%outChain_0, %buf0, %buf1)
      %outChain_2 = use(%outChain_1, %buf1, %buf0)
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DynamicMemoryPlanningPass
//===----------------------------------------------------------------------===//

def DynamicMemoryPlanningPass : Pass<"mgp-dynamic-memory-planning", "MGP::ModelOp"> {
  let summary = "Dynamic (runtime) memory planning";
  let description = [{
    Uses BufferLifetimeAnalysisPass to compute lifetimes of all buffers, then
    inserts BufferPlanOp to do runtime memory planning. The high watermark
    returned by BufferPlanOp is used to make a single large allocation, and all
    former allocations are replaced with slices (BufferSliceOp) into the single
    large allocation.
  }];
}

//===----------------------------------------------------------------------===//
// AttachPythonStackTraces
//===----------------------------------------------------------------------===//

def AttachPythonStackTraces : Pass<"mgp-attach-python-stack-traces", "mlir::ModuleOp"> {
  let summary = "Attaches Python stack traces from OpaqueLoc to ops and asserts";
  let description = [{
    Walks all ops in the module. For each op with an OpaqueLoc wrapping a
    PythonFrameAttr, encodes the Python stack trace as FusedLoc metadata
    (so it survives downstream lowering) and strips the OpaqueLoc. Also
    embeds the trace directly into the message of any mgp.assert op for
    better runtime diagnostics.
  }];
}

//===----------------------------------------------------------------------===//
// GlobalizeSlotIndicesPass
//===----------------------------------------------------------------------===//

def GlobalizeSlotIndicesPass : Pass<"mgp-globalize-slot-indices", "mlir::ModuleOp"> {
    let summary = "Globalizes slot indices across all models in the module";
    let description = [{
     Gives each `mgp.model`, in definition order, a unique slot index range
     offset by the slots of preceding models, and sets `num_slots` of *all*
     `mgp.context.create` ops to the module total. This prevents slot conflicts
     when models are combined or inlined later.

     Example:

     BEFORE:
     ```mlir
     module {
       mgp.model @sub1 init {
         %ctx = mgp.context.create 2 // Uses slots 0, 1
         %rt = mgp.runtime.create %chain, %ctx[0]
         %c = mgp.buffer.cache %chain, %ctx[1], %buf1
         ...
       } execute {
         %b = mgp.buffer.get_cached %chain, %ctx[1]
         ...
       }
       mgp.model @sub2 init {
         %ctx = mgp.context.create 1 // Uses slot 0
         %rt = mgp.runtime.create %chain, %ctx[0]
         ...
       } execute { ... }
     }
     ```

     AFTER:
     ```mlir
     module {
       mgp.model @sub1 init {
         // Total slots updated to 3 (2 from @sub1 + 1 from @sub2)
         %ctx = mgp.context.create 3
         // Slots for @sub1 start at 0 (no offset)
         %rt = mgp.runtime.create %chain, %ctx[0]
         %c = mgp.buffer.cache %chain, %ctx[1], %buf1
         ...
       } execute {
         %b = mgp.buffer.get_cached %chain, %ctx[1]
         ...
       }
       mgp.model @sub2 init {
         // Total slots updated to 3
         %ctx = mgp.context.create 3
         // Slots for @sub2 start at 2 (offset by @sub1's 2 slots)
         %rt = mgp.runtime.create %chain, %ctx[2]
         ...
       } execute { ... }
     }
     ```
    }];
}

//===----------------------------------------------------------------------===//
// InlineMGPCallsPass
//===----------------------------------------------------------------------===//

def InlineMGPCallsPass : Pass<"mgp-inline-mgp-calls", "mlir::ModuleOp"> {
  let summary = "Inlines the `execute` block of subgraphs at `mgp.call` sites";
  let description = [{
    Visits the `mgp.model` call graph in post-order so callees are inlined
    before callers. Recursion is an error.

    For each `mgp.call` in the caller's `init` and `execute` regions:
    1.  Once per callee, the callee's `init` ops (excluding `mgp.context.create`
        and the terminator) are cloned before the caller's `init` terminator,
        mapping the weights registry, devices and context.
    2.  The callee's `execute` ops (excluding the terminator) are cloned at the
        call site, mapping call operands to block arguments and results back.
    3.  The `mgp.call` is replaced with the inlined results and erased.

    Inlined `mgp.model`s left without uses are then erased.
  }];
}

//===----------------------------------------------------------------------===//
// OutlineModelsToCallables
//===----------------------------------------------------------------------===//

def OutlineModelsToCallables : Pass<"mgp-outline-models-to-callables", "mlir::ModuleOp"> {
  let summary = "Converts all MGP::ModelOps into component MEF::CallableOps and wrapping MEF::ModelOps";
  let description = [{
    Lowers all `MGP::ModelOps`, including subgraphs, into a structure directly
    executable by GraphRT, in two steps:
    1) Propagate prefix handling through the model call graph.
    2) Lower the denormalized `mgp.model` ops to:
        a) `mef.model` entry points, packaging the init and exec regions for
           the executor, and
        b) `mef.callable` ops holding the model's component regions, callable
           by `grt.invoke` from other regions.

    Subgraphs are thus reused at runtime across invocations rather than
    inlined and monomorphised.

    `mef.model` only requires that init's output be a prefix of exec's input,
    with the rest of exec's input matching the model's input signature.

    Example:

    BEFORE:
    ```mlir
    module {
      mgp.model @exp subgraph : mof ...
      init (/* weights registry, device contexts */...) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 2
        %2 = mgp.runtime.create %0, %1[0]
        %3 = mgp.module.load %2, %1[0] [0], ...
        %4 = mgp.string.create "foo" : !mgp.opaque<"mojo_string">
        %5 = mgp.buffer.constant.external %3, %arg0[%4, 80, align 16 placeholder true] ...
        %6 = mgp.buffer.cache %0, %1[1], %5
        mef.output %6, %1, %arg0, %arg1
      } execute (/* implicits */..., %arg4: !mgp.tensor<4x5xf32>, %arg5: index)
      -> (!mef.chain, !mgp.tensor<4x5xf32>) {
        ...
        mef.output %arg0, %3 : !mef.chain, !mgp.tensor<4x5xf32>
      }
      mgp.model @main : mof
        {argument_device_indices = [0 : ui32], result_device_indices = [0 : ui32]}
        devices [#M.device_ref<"cpu", 0>]
        init (/* weights registry, device contexts */)
      {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 1
        %2 = mgp.runtime.create %0, %1[0]
        %3 = mgp.module.load %2, %1[0] [0], ...
        mef.output %3, %1, %arg0, %arg1
      } execute (/* implicits */..., %arg4: !mgp.tensor<4x5xf32>)
      -> (!mef.chain, !mgp.tensor<4x5xf32>) {
        ...
        %1:2 = mgp.call @exp(%arg0, %arg1, %arg2, %arg3, %arg4, %0) {..., prefix = "prefix"} ...
        mef.output %1#0, %1#1 : !mef.chain, !mgp.tensor<4x5xf32>
      }
    }
    ```

    AFTER:
    ```mlir
    module {
      mef.callable @exp$init(..., %arg2: !mgp.opaque<"mojo_string">) -> (
        !mef.chain,
        !mgp.context,
        !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"cpu", 0>>,
        !mgp.opaque<"mojo_string">
      ) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 2
        %2 = mgp.runtime.create %0, %1[0] // note that subgraphs still
                                          // get their own context and runtime
        %3 = mgp.module.load %2, %1[0] [0], ...
        %4 = mgp.string.create "foo"
        %5 = mgp.string.concat %arg2, %4 // NEW
        %6 = mgp.buffer.constant.external %3,
            %arg0[%5, 80, align 16 placeholder true]
            // MODIFIED: uses concat'd prefix from %5
        %7 = mgp.buffer.cache %0, %1[1], %6
        mef.output %7, %1, %arg0, %arg1, %arg2
              : !mef.chain,
                !mgp.context,
                !mgp.opaque<"weights_registry">,
                !mgp.device_context<<"cpu", 0>>,
                !mgp.opaque<"mojo_string"> // added context to pass to exec
      }
      mef.callable @exp$exec(/* implicits */...,
        %arg4: !mgp.opaque<"mojo_string">,
        %arg5: !mgp.tensor<4x5xf32>,
        %arg6: index)
      -> (!mef.chain, !mgp.tensor<4x5xf32>) {
        ...
        mef.output %arg0, %3 : !mef.chain, !mgp.tensor<4x5xf32>
      }
      mef.callable @main$init(
        %arg0: !mgp.opaque<"weights_registry">,
        %arg1: !mgp.device_context<<"cpu", 0>>)
      -> (...) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 1
        %2 = mgp.runtime.create %0, %1[0]
        %3 = mgp.module.load %2, %1[0] [0], ...
        %4 = mgp.string.create "prefix" : !mgp.opaque<"mojo_string">
        %5:5 = grt.invoke @exp$init(%arg0, %arg1, %4)
        mef.output %3, %1, %arg0, %arg1, %5#0, %5#1, %5#2, %5#3, %5#4
            : !mef.chain,
              !mgp.context,
              !mgp.opaque<"weights_registry">,
              !mgp.device_context<<"cpu", 0>>, // main's forwarded implicits
              !mef.chain,
              !mgp.context,
              !mgp.opaque<"weights_registry">,
              !mgp.device_context<<"cpu", 0>>, // exp's forwarded implicits
              !mgp.opaque<"mojo_string"> // exp's SSA prefix for all placeholder
                                         // accesses to weights registry
      }
      mef.callable @main$exec(
        /* main's forwarded implicits */...,
        /* subgraph's forwarded implicits */...,
        %arg9: !mgp.tensor<4x5xf32>)
      -> (...) {
        %0 = "mgp.generic.execute"(%arg1) ...
        %1:2 = grt.invoke @exp$exec(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %0) ...
        mef.output %1#0, %1#1 : !mef.chain, !mgp.tensor<4x5xf32>
      }
      mef.model @main : mof ... {
        %0:9 = grt.invoke @main$init(%arg0, %arg1) : ...
        mef.output %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8 : ...
      } execute (...) -> (...) {
        %0:2 = grt.invoke @main$exec(
          arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : ...
        mef.output %0#0, %0#1 ...
      }
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StageDeviceGraphsPass
//===----------------------------------------------------------------------===//

def StageDeviceGraphsPass :
    Pass<"mgp-stage-device-graphs", "MGP::ModelOp"> {
  let summary = "Stage the execute body of an `isDeviceGraph` model into a "
                "`mgp.device_graph.staging` op.";
  let description = [{
    Part of device-graph synthesis. For each `mgp.model` marked `isDeviceGraph`,
    moves every `mogg.routine.async` in the execute block into a single
    `mgp.device_graph.staging` op. The regions' free values become the op's
    inputs (entering the body as `!mogg.async<T>` block args, unpacked inside),
    and values produced inside but consumed outside become the op's results
    (packed for the async yield, external uses rewired). The staging op is an
    `IsolatedFromAbove`, transient container that later synthesis passes lower
    into `mgp.device_graph.create` / `mgp.device_graph.execute`.

    For models without `isDeviceGraph`, the pass is a no-op.
  }];
}

//===----------------------------------------------------------------------===//
// PartitionDeviceGraphCollectivePass
//===----------------------------------------------------------------------===//

def PartitionDeviceGraphCollectivePass :
    Pass<"mgp-partition-device-graph-collective", "MGP::ModelOp"> {
  let summary = "Partition a `mgp.device_graph.staging` op into per-device "
                "`mgp.device_graph.collective` regions.";
  let description = [{
    Part of device-graph synthesis; runs after `mgp-stage-device-graphs`. For
    each `mgp.model` marked `isDeviceGraph`, converts the single
    `mgp.device_graph.staging` op in the execute block into a
    `mgp.device_graph.collective` op with one region per device. Cross-device
    values are exchanged through stable address slots, declared on the
    collective's `stableTypes` and reached by index through the region's
    builder with `mgp.device_graph.stable.place` and
    `mgp.device_graph.stable.get`.

    Each `mgp.device_graph.stable.create` is lowered locally: the producing
    region publishes its tensor with `stable.place`, every other region that
    consumes it reads the slot with `stable.get`, and in both cases
    `mgp.device_graph.stable.tensor` rebuilds the `!mgp.tensor<..., stable>`
    the consumers were lowered against, so the compiled regions are cloned
    unchanged. The producing region takes the tensor's spec from the tensor
    itself; a remote region has only the slot, so its spec comes from the
    static type, and a stable tensor with dynamic dimensions cannot cross
    devices yet.

    For models without `isDeviceGraph` or without a staging op, the pass is a
    no-op.
  }];
}

//===----------------------------------------------------------------------===//
// ComputeDeviceGraphDependenciesPass
//===----------------------------------------------------------------------===//

def ComputeDeviceGraphDependenciesPass :
    Pass<"mgp-compute-device-graph-dependencies", "MGP::ModelOp"> {
  let summary = "Convert the staged compiled regions of an `isDeviceGraph` "
                "model into `mgp.device_graph.dependencies` ops.";
  let description = [{
    Part of device-graph synthesis; runs after `mgp-stage-device-graphs`. For
    each `mgp.model` marked `isDeviceGraph`, converts every
    `mogg.routine.async` inside the model's `mgp.device_graph.staging`
    op into a `mgp.device_graph.dependencies` op: the region body moves in and
    references enclosing SSA directly (the dataflow boundary dissolves, with
    `mogg.async.pack` / `mogg.async.unpack` inserted to keep the verbatim body's
    types consistent), and the dependency edges are derived from the region's
    chain and tensor inputs. The conversion is 1:1 and local; the edge set is
    complete at this point (no later stage adds an edge).

    For models without `isDeviceGraph`, the pass is a no-op.
  }];
}

//===----------------------------------------------------------------------===//
// SimplifyDeviceGraphDependenciesPass
//===----------------------------------------------------------------------===//

def SimplifyDeviceGraphDependenciesPass :
    Pass<"mgp-simplify-device-graph-dependencies", "MGP::ModelOp"> {
  let summary = "Remove redundant dependency edges from "
                "`mgp.device_graph.dependencies` ops.";
  let description = [{
    Part of device-graph synthesis; runs after
    `mgp-compute-device-graph-dependencies`, which derives each op's edge set
    locally (one edge per producing region of any operand) and so records
    edges already implied transitively. Given:
    ```mlir
      %node_b, ... = mgp.device_graph.dependencies ... dependencies(%node_c) ...
      %node_a, ... = mgp.device_graph.dependencies ... dependencies(%node_b, %node_c) ...
    ```
    the direct `%node_c` edge on `%node_a` is redundant: a dependencies op's
    node result transitively gates on its own edges (even for an empty body),
    so `%node_b` already orders `%node_a` after `%node_c`. Such edges are
    removed:
    ```mlir
      %node_a, ... = mgp.device_graph.dependencies ... dependencies(%node_b) ...
    ```
    Each surviving edge becomes an entry in the `dependencies=[...]` list of
    the emitted `DeviceGraphBuilder.region(...)` call, so fewer edges means
    less host-side work per graph build.
  }];

  let statistics = [
    Statistic<"numEliminatedDependencies", "num-eliminated-dependencies",
              "the number of redundant dependency edges eliminated">,
  ];
}

//===----------------------------------------------------------------------===//
// GenerateBuildGraphStubsPass
//===----------------------------------------------------------------------===//

def GenerateBuildGraphStubsPass :
    Pass<"mgp-generate-build-graph-stubs", "mlir::ModuleOp"> {
  let summary = "Generate `@stub_N_build_graph` recording stubs and retarget the "
                "staged kernel-stub calls of `isDeviceGraph` models to them.";
  let description = [{
    Part of device-graph synthesis; runs after
    `mgp-compute-device-graph-dependencies`. Runs on the module so it can add
    module-scope recording stubs. For each `mgp.model` marked
    `isDeviceGraph`, finds every `mogg.call @stub_N` inside the
    `mgp.device_graph.staging` op of the model's execute block and, for each
    distinct callee, generates a sibling recording stub `@stub_N_build_graph`.
    The sibling is an ordinary `mogg.func` whose producer dispatch records
    into a `!mgp.device_graph.builder` (via `mogg.call.build_graph`) instead of
    dispatching (`mogg.call.execute`), and which yields an
    `!mgp.device_graph.node` handle. The staged call is then retargeted to invoke
    the sibling stub.

    Scope is limited to the staged execute body: stub calls elsewhere in the
    model (e.g. weight-prepack stubs that EICM lifts into the `init` block) are
    left as direct dispatches.

    For models without `isDeviceGraph`, the pass is a no-op.
  }];
}

//===----------------------------------------------------------------------===//
// FinalizeDeviceGraphsPass
//===----------------------------------------------------------------------===//

def FinalizeDeviceGraphsPass :
    Pass<"mgp-finalize-device-graphs", "MGP::ModelOp"> {
  let summary = "Lower the `mgp.device_graph.staging` op of an `isDeviceGraph` "
                "model into `mgp.device_graph.create` / `mgp.device_graph.execute`.";
  let description = [{
    Final stage of device-graph synthesis; runs after
    `mgp-generate-build-graph-stubs`. For each `mgp.model` marked `isDeviceGraph`,
    lifts the (cleaned) `mgp.device_graph.staging` op in the execute block into
    the device-graph surface:

    - the staged body becomes the build body of a `mgp.device_graph.create` --
      its `!mgp.device_graph.builder` and bare-`T` input block args carry over,
      and the `staging.yield`'s escaping values become the
      `mgp.device_graph.yield`'s operands (defining the result graph signature's
      results);
    - a `mgp.device_graph.execute` replays the graph against the same inputs and
      surfaces its outputs as results.

    `mef.output` is rewired to the execute op's chain and output results, and the
    staging op is erased. The create op draws its chain / context / device
    context from the execute block (`mef.output`'s chain operand, the model
    context, the model device context).

    First cut: single device per model, no caching (a reserved slot 0).

    For models without `isDeviceGraph`, the pass is a no-op.
  }];

  // The pass wraps `mgp.device_graph.create` in a `mogg.routine.async`
  // (MOGG ops/attrs) carrying a `KGEN::ParamDeclArrayAttr`, so those dialects
  // must be loaded even for inputs that don't already reference them.
  let dependentDialects = ["M::MOGG::MOGGDialect", "M::KGEN::KGENDialect"];
}

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
