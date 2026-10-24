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
//
// This defines the device-graph operations in the MGP dialect: the
// `mgp.device_graph.create`/`mgp.device_graph.execute` surface, its
// `mgp.device_graph.yield` build-body terminator, and the transient
// `mgp.device_graph.staging`/`mgp.device_graph.dependencies` scaffolds (with
// their yield terminators) used during device-graph synthesis.
//
// This file is included by MGPOps.td; the ops below use the local
// `MGP_DeviceGraphOp` base class, so they do not depend on `MGP_Op`.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPDEVICEGRAPHOPS_TD
#define GENERICML_MGPDIALECT_MGPDEVICEGRAPHOPS_TD

include "GraphCompiler/Interfaces/EmitsMojoInterface.td"
include "GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "GraphCompiler/MGPDialect/Dialect/MGPTypes.td"
include "GraphCompiler/MOGGDialect/MOGGTypes.td"
include "MLRT/GraphRT/GRTDialect/GRTMayAliasInterface.td"
include "MLRT/GraphRT/MEFDialect/MEFTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for device-graph ops in the MGP dialect. Prepends `device_graph.`
// to the mnemonic, so `MGP_DeviceGraphOp<"create">` yields the op name
// `mgp.device_graph.create`.
class MGP_DeviceGraphOp<string mnemonic, list<Trait> traits = []> :
  Op<MGPDialect, "device_graph." # mnemonic, traits>;

//===----------------------------------------------------------------------===//
// DeviceGraphCacheOp
//===----------------------------------------------------------------------===//

def DeviceGraphCacheOp : MGP_DeviceGraphOp<"cache", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  Pure,
]> {
  let summary = "Returns a model's device-graph cache from its state context.";
  let description = [{
    Returns the `!mgp.device_graph.cache` held by the `!mgp.context`, for
    `mgp.device_graph.create` to reuse graphs across executions. The cache is
    created on first use and lives as long as the state context, so it is shared
    by every execution of the model.

    Example:
    ```mlir
      %cache = mgp.device_graph.cache %ctx : !mgp.device_graph.cache
    ```
  }];

  let arguments = (ins MGP_ContextType:$theContext);
  let results = (outs MGP_DeviceGraphCacheType:$outCache);

  let assemblyFormat = [{
    $theContext attr-dict `:` qualified(type($outCache))
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphYieldOp : MGP_DeviceGraphOp<"yield", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  HasParent<"DeviceGraphCollectiveCreateOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a device-graph build body.";
  let description = [{
    Marks the end of the recording build body of an enclosing
    `mgp.device_graph.create` or `mgp.device_graph.collective.create`, yielding
    the graph's output values. The operand types form the result
    `!mgp.device_graph` signature's results; the parent op instantiates the
    recorded builder into that graph after the body returns.

    Example:
    ```mlir
      mgp.device_graph.yield %out : !mgp.buffer<<"gpu", 0>>
    ```
  }];

  // A device graph's outputs are the device-side values it produces: host-side
  // ordering chains are stripped from the signature at finalization (see
  // `FinalizeDeviceGraphsPass`), so only buffers and tensors reach this yield.
  let arguments = (ins
    Variadic<AnyTypeOf<[MGP_DeviceBufferType, MGP_DeviceTensorType]>>:$outputs);
  let assemblyFormat = "attr-dict ($outputs^ `:` type($outputs))?";

  // No-operand builder for the parent's `SingleBlockImplicitTerminator` trait
  // (operands are populated after insertion); see DeviceGraphStagingYieldOp.
  let builders = [OpBuilder<(ins)>];
}

//===----------------------------------------------------------------------===//
// DeviceGraphAllocOp
//===----------------------------------------------------------------------===//

def DeviceGraphAllocOp : MGP_DeviceGraphOp<"alloc", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Allocates a buffer against a device-graph builder.";
  let description = [{
    The device-graph counterpart of `mgp.buffer.alloc`: allocates an
    uninitialized buffer of `runtimeSize` bytes, but takes the recording
    `!mgp.device_graph.builder` instead of a `!mgp.device_context`. Keeping
    allocation a builder operation lets the builder track device-graph
    allocations (e.g. a device-graph memory pool) in the future; today it lowers
    to a Mojo primitive that allocates via the builder's device context.

    It appears only inside a `mgp.device_graph.create` build body and is emitted
    to Mojo (never C++-lowered).

    Example:
    ```mlir
      %buf = mgp.device_graph.alloc %ch, %size, %builder, align 256
          : !mgp.buffer<<"gpu", 0>>
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    Index:$runtimeSize,
    MGP_DeviceGraphBuilderType:$graphBuilder,
    DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign
  );
  let results = (outs MGP_DeviceBufferType:$outBuffer);

  let assemblyFormat = [{
    $inChain `,` $runtimeSize `,` $graphBuilder ( `,` `align` $rawAlign^ )?
    attr-dict `:` qualified(type($outBuffer))
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphRecordingContextOp
//===----------------------------------------------------------------------===//

def DeviceGraphRecordingContextOp : MGP_DeviceGraphOp<"recording_context", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Views a device-graph builder as a recording device context.";
  let description = [{
    Extracts a recording `!mogg.context` from a `!mgp.device_graph.builder`.
    Operations dispatched against the returned context record into the builder
    (in enqueue order) instead of executing eagerly.

    This is how an execute-only kernel -- one without a node-free `build_graph`
    method -- is recorded during device-graph synthesis: the recording stub
    obtains this context and passes it as the launch argument of the kernel's
    `mogg.call.execute`, so the kernel's ordinary `execute` dispatches record
    into the graph. It appears only inside a device-graph build body and is
    emitted to Mojo (never C++-lowered), lowering to
    `DeviceGraphBuilder.recording_context()`.

    Example:
    ```mlir
      %ctx = mgp.device_graph.recording_context %builder : !mogg.context
    ```
  }];

  let arguments = (ins MGP_DeviceGraphBuilderType:$graphBuilder);
  let results = (outs MOGG_Context:$context);

  let assemblyFormat = "$graphBuilder attr-dict `:` qualified(type($context))";
}

//===----------------------------------------------------------------------===//
// DeviceGraphDeviceContextOp
//===----------------------------------------------------------------------===//

def DeviceGraphDeviceContextOp : MGP_DeviceGraphOp<"device_context", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Extracts the device context backing a device-graph builder.";
  let description = [{
    Reads the `!mgp.device_context` a `!mgp.device_graph.builder` was created
    from, lowering to `DeviceGraphBuilder.context()`.

    This is the *backing* context, not a recording view: work dispatched against
    it runs eagerly while the graph is being built instead of being recorded as
    a graph node. It is therefore only for in-body ops that need a real device
    context to construct or register a host-side view of device memory -- e.g.
    `mgp.buffer.bulk_slice`, which uses it for sub-buffer pointer math and
    pointer registration and enqueues nothing. An op that must record its work
    takes a `mgp.device_graph.recording_context` instead.

    Example:
    ```mlir
      %ctx = mgp.device_graph.device_context %builder
          : !mgp.device_context<<"gpu", 0>>
    ```
  }];

  let arguments = (ins MGP_DeviceGraphBuilderType:$graphBuilder);
  let results = (outs MGP_DeviceContextType:$deviceContext);

  let assemblyFormat =
    "$graphBuilder attr-dict `:` qualified(type($deviceContext))";
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesOp
//===----------------------------------------------------------------------===//

def DeviceGraphDependenciesOp : MGP_DeviceGraphOp<"dependencies", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  HasOnlyGraphRegion,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"DeviceGraphDependenciesYieldOp">,
]> {
  let summary = "A compiled region with explicit device-graph dependency edges.";
  let description = [{
    One is created per `mogg.routine.async` when lowering towards
    device-graph execution. It separates the region's dataflow from its
    ordering, so region bodies can be merged while their execution order
    dependencies are still tracked.

    - It is **not** `IsolatedFromAbove`: the body references enclosing SSA
      values directly (the region's dataflow boundary is dissolved at
      conversion time). Execution order is encoded only by the `dependencies`
      operands.
    - The `graphBuilder` operand is the enclosing builder (the
      `mgp.device_graph.staging` / `mgp.device_graph.create` builder), making
      the builder dependency explicit in the def-use graph.
    - The body has exactly one block argument, a `!mgp.device_graph.builder`
      bound to `graphBuilder`. The body's device-context uses are re-sourced
      from it via a `builtin.unrealized_conversion_cast`, so the body is
      self-contained (local lowering).
    - The `dependencies` operands are only the dependency edges:
      `!mgp.device_graph.node` values of the units this one must run after.
    - It produces a `!mgp.device_graph.node` for this unit (successors depend
      on it), synthesized by the op rather than yielded, plus the former
      region's results (so existing users are untouched).
    - The body's nodes are recorded in order: `DeviceGraphBuilder.region`
      chains each node after the previous one and the `dependencies` edges
      gate the first, so the node result is the body's last node.

    The body terminates with `mgp.device_graph.dependencies.yield`, whose operand
    types must match the op's `bodyResults`.

    Example:
    ```mlir
      %node, %r = mgp.device_graph.dependencies builder(%b)
          dependencies(%depA, %depB) {
      ^bb0(%builder: !mgp.device_graph.builder):
        // ops copied from the former mogg.routine.async, referencing
        // enclosing SSA directly
        mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
      } : (!mgp.device_graph.node, !mgp.device_graph.node)
        -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)
    ```
  }];

  let arguments = (ins
    MGP_DeviceGraphBuilderType:$graphBuilder,
    Variadic<MGP_DeviceGraphNodeType>:$dependencies
  );

  let results = (outs
    MGP_DeviceGraphNodeType:$node,
    Variadic<AnyType>:$bodyResults
  );

  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins
      "Value":$graphBuilder,
      "ValueRange":$dependencies,
      CArg<"function_ref<void(OpBuilder &, Location, Value)>",
           "nullptr">:$bodyBuilder)>
  ];

  let assemblyFormat = [{
    `builder` `(` $graphBuilder `)`
    `dependencies` `(` ($dependencies^)? `)`
    $body attr-dict (`:` type($bodyResults)^)?
  }];

  let extraClassDeclaration = [{
    /// Returns the `mgp.device_graph.dependencies.yield` terminating the body.
    DeviceGraphDependenciesYieldOp getYieldOp();

    /// Returns the single body region block.
    Block &getBodyBlock();

    /// Returns the body's `!mgp.device_graph.builder` block argument (bound to
    /// the `graphBuilder` operand).
    TypedValue<MGP::DeviceGraphBuilderType> getBuilderBlockArgument();

    /// Returns true if the body _may_ enqueue new device graph nodes. Best
    /// effort, based on the uses of the body's builder.
    bool producesNewNodes();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphDependenciesYieldOp : MGP_DeviceGraphOp<"dependencies.yield", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  HasParent<"DeviceGraphDependenciesOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a `mgp.device_graph.dependencies` body.";
  let description = [{
    Yields the former region's result values out of an enclosing
    `mgp.device_graph.dependencies` op. The operand types must match the parent
    op's `bodyResults`.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  // No-operand builder for the `SingleBlockImplicitTerminator` trait on the
  // parent, which inserts a default terminator into an empty body; the operands
  // are populated afterward.
  let builders = [OpBuilder<(ins)>];
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingOp
//===----------------------------------------------------------------------===//

def DeviceGraphStagingOp : MGP_DeviceGraphOp<"staging", [
  HasOnlyGraphRegion,
  IsolatedFromAbove,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"DeviceGraphStagingYieldOp">,
]> {
  let summary = "Container operation for staging the device-graph build body.";
  let description = [{
    Holds the device-graph build body while `--mgp-stage-device-graphs`
    assembles it: the `mgp.device_graph.dependencies` ops plus the
    `mogg.async` pack/unpack ops mediating the `!mgp.*`/`!mogg.async<...>` and
    `!mo.tensor` type domains. It is eliminated when its contents are lifted
    into `mgp.device_graph.create`, so it never reaches EmitMojo.

    Like `mogg.routine.async` it is `IsolatedFromAbove`: `inputs`
    operands of type `T` enter as block arguments of the same bare `T`, and the
    body yields bare `T` values typed as `stagedResults`. The verifier requires
    these types to match exactly; no `!mogg.async<T>` crosses the boundary.
    `mgp-compute-device-graph-dependencies` injects the async pack/unpack
    locally when it lowers each region.

    The first block argument is the `!mgp.device_graph.builder` (with no
    corresponding operand), which maps onto the `mgp.device_graph.create`
    builder at finalization. It remains a graph region, since the staged
    dependencies ops carry node edges.

    Example:
    ```mlir
      %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
      ^bb0(%builder: !mgp.device_graph.builder,
           %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
        // dependencies ops (each injecting its own async pack/unpack mediation)
        mgp.device_graph.staging.yield %t : !mgp.tensor<2x3xf32, <"gpu", 0>>
      } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    // Constant resources of the inlined mogg.routine.async ops; this
    // op is later synthesized back into one.
    DefaultValuedAttr<ArrayAttr, "{}">:$constantResources
  );

  let results = (outs Variadic<AnyType>:$stagedResults);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    // Builds the op and its body in one shot: adds the `inputs` operands, creates
    // the body block with its `!mgp.device_graph.builder` block argument followed
    // by one `!mogg.async<T>` argument per input, and invokes `bodyBuilder` with
    // the builder arg and the async input args. The callback is expected to emit
    // the `staging.yield` carrying the (async-wrapped) staged values; the op's
    // result types are inferred by unwrapping those yield operand types. If
    // `bodyBuilder` is null or leaves the body unterminated, an operandless
    // `staging.yield` (zero results) is supplied.
    OpBuilder<(ins
      "ValueRange":$inputs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let assemblyFormat = [{
    `inputs` `(` ($inputs^ `:` qualified(type($inputs)))? `)`
    (`constantResources` `=` $constantResources^)?
    $body attr-dict (`->` qualified(type($stagedResults))^)?
  }];

  let extraClassDeclaration = [{
    /// Returns the `mgp.device_graph.staging.yield` terminating the body.
    DeviceGraphStagingYieldOp getYieldOp();

    Block &getBodyBlock() {
      return getBody().front();
    }

    TypedValue<DeviceGraphBuilderType> getBodyBuilder() {
      return cast<TypedValue<DeviceGraphBuilderType>>(
        getBodyBlock().getArgument(0));
    }

    MutableArrayRef<BlockArgument> getBodyInputs() {
      return getBodyBlock().getArguments().drop_front(1);
    }
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphStagingYieldOp : MGP_DeviceGraphOp<"staging.yield", [
  HasParent<"DeviceGraphStagingOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a `mgp.device_graph.staging` body.";
  let description = [{
    Yields the values the rest of `execute` consumes out of an enclosing
    `mgp.device_graph.staging` op. The operand types must match the parent op's
    `stagedResults`.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  // No-operand builder for the `SingleBlockImplicitTerminator` trait on the
  // parent (see DeviceGraphDependenciesYieldOp).
  let builders = [OpBuilder<(ins)>];
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableCreateOp
//===----------------------------------------------------------------------===//

def DeviceGraphStableCreateOp : MGP_DeviceGraphOp<"stable.create", [
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Marks a tensor as requiring a stable address.";
  let description = [{
    Designates `$tensor` as a value that must be assigned a stable address when
    the device graph is partitioned across devices. The result has the same
    shape, device, and mutability as `$tensor`, but its `stable` flag is set.
    The result aliases `$tensor`: the slot holds its data pointer, and every
    device region of the partitioned graph may read through it.

    `$runtimeDims` carries one `index` value per dynamic dimension of
    `$tensor`, in dimension order. The slot holds only a data pointer, and the
    tensor that knows its runtime shape may live in another device's region
    of the partitioned graph, so a consumer rebuilds the shape from these
    values instead. The MO lowering populates them from the graph's symbolic
    dimension parameters, which every region can see.

    Example:
    ```mlir
      %shared = mgp.device_graph.stable.create %t
          : !mgp.tensor<4xf32, <"gpu", 0>>
      %dynamic = mgp.device_graph.stable.create %u[%n]
          : !mgp.tensor<?x3xf32, <"gpu", 0>>
    ```
  }];

  let arguments = (ins
    MGP_DeviceTensorType:$tensor,
    Variadic<Index>:$runtimeDims
  );
  let results = (outs MGP_DeviceTensorType:$result);

  let assemblyFormat = [{
    $tensor (`[` $runtimeDims^ `]`)? attr-dict `:` qualified(type($tensor))
  }];

  let extraClassDeclaration = [{
    /// Get the stable address type that will back this tensor.
    MGP::StableAddressType getStableType() {
      return MGP::StableAddressType::get(getTensor().getType());
    }

    /// The device for the stable tensor.
    M::DeviceRefAttr getDevice() {
      return getTensor().getType().getDevice();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableGetOp
//===----------------------------------------------------------------------===//

def DeviceGraphStableGetOp : MGP_DeviceGraphOp<"stable.get", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Reads a stable address slot in a device graph region.";
  let description = [{
    Returns stable address slot `$index` of the enclosing collective for the
    region's consumers, reading it through `$graphBuilder`, which holds the
    collective's slot table. `$index` names an entry of the collective's
    `stableTypes`, and the result type must equal that entry. The op is
    verified against the nearest enclosing `mgp.device_graph.collective` or,
    after finalization, `mgp.device_graph.collective.create`; lowering may
    wrap it in intermediate regions on the way there.

    The result is the slot itself, not a tensor. The slot holds a data pointer
    that `mgp.device_graph.stable.place` rewrites before every replay, so a
    tensor built over the data pointer would bake a stale address into the
    graph. `mgp.device_graph.stable.tensor` turns the slot back into the
    `!mgp.tensor<..., stable>` consumers were lowered against, whose data
    pointer is the slot address; the kernel then reads through it via a
    `stable`-marked `!mo.tensor` built by `mogg.tensor.__init__`.

    The result is a `stable` `!mgp.device_tensor` whose shape/device come from
    the slot's element type `T`, since the slot provides its backing address.

    Example:
    ```mlir
      %a1 = mgp.device_graph.stable.get %b[1]
          : !mgp.stable_addr<<4xf32, <"gpu", 1>>>
    ```
  }];

  let arguments = (ins
    MGP_DeviceGraphBuilderType:$graphBuilder,
    I64Attr:$index
  );
  let results = (outs MGP_StableAddressType:$result);
  let assemblyFormat = [{
    $graphBuilder `[` $index `]` attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphStablePlaceOp
//===----------------------------------------------------------------------===//

def DeviceGraphStablePlaceOp : MGP_DeviceGraphOp<"stable.place", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  // Relating the result to `$tensor` also lets ODS infer the result type.
  TypesMatchWith<"stable address type wraps the tensor type",
                 "tensor", "placedAddr",
                 "::M::MGP::StableAddressType::get($_self)">,
]> {
  let summary = "Publishes a value into a stable address slot.";
  let description = [{
    Records that the enclosing region publishes `$tensor` into stable address
    slot `$index` of the enclosing collective, through `$graphBuilder`, which
    holds the collective's slot table. At replay the slot is rewritten to hold
    `$tensor`'s data pointer before the graphs that read it run. `$index`
    names an entry of the collective's `stableTypes`; that entry must be the
    stable address type of `$tensor`, checked against the nearest enclosing
    collective as for `mgp.device_graph.stable.get`.

    The result is the slot, sequenced after the publish. MGP conveys ordering
    through dataflow, so consumers on the source device take the placed slot
    and need no separate `mgp.device_graph.stable.get`. Like that op's result,
    it is the slot rather than a tensor view; see `mgp.device_graph.stable.get`
    for why.

    Example:
    ```mlir
      %p0 = mgp.device_graph.stable.place %b[0], %t
          : !mgp.tensor<4xf32, <"gpu", 0>>
    ```
  }];

  let arguments = (ins
    MGP_DeviceGraphBuilderType:$graphBuilder,
    MGP_DeviceTensorType:$tensor,
    I64Attr:$index
  );
  let results = (outs MGP_StableAddressType:$placedAddr);
  let assemblyFormat = [{
    $graphBuilder `[` $index `]` `,` $tensor attr-dict `:` qualified(type($tensor))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableTensorOp
//===----------------------------------------------------------------------===//

def DeviceGraphStableTensorOp : MGP_DeviceGraphOp<"stable.tensor", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  InferTypeOpAdaptor,
  Pure,
  // Also lets the declarative format omit the spec's type.
  TypesMatchWith<"spec matches the slot's tensor spec", "addr", "spec",
                 "::llvm::cast<::M::MGP::StableAddressType>($_self)"
                 ".getElementType().getTensorSpec()">,
]> {
  let summary = "Views a stable address slot as the stable tensor it holds.";
  let description = [{
    Builds the `!mgp.tensor<..., stable>` a slot stands for: `$addr`'s element
    type with its `stable` flag set, shaped by `$spec`. The tensor's data
    pointer is the slot's address rather than the pointer the slot holds, so a
    kernel recorded against it extracts the buffer as for any other tensor and
    dereferences the slot when it runs, through a `stable`-marked `!mo.tensor`
    built by `mogg.tensor.__init__`, instead of baking in whatever the slot
    pointed at during recording.

    `mgp-partition-device-graph-collective` uses this to reproduce a
    `mgp.device_graph.stable.create` result inside each collective region:
    `stable.place` or `stable.get` yields the slot and this op turns it back
    into the tensor the region's consumers were lowered against. The slot
    carries only the data pointer, so `$spec` supplies the shape; it must be
    the slot element type's spec.

    Example:
    ```mlir
      %a0 = mgp.device_graph.stable.get %b[0]
          : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
      %spec = mgp.tensor_spec.create : !mgp.tensor_spec<4xf32>
      %t = mgp.device_graph.stable.tensor %a0, %spec
          : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
      // %t : !mgp.tensor<4xf32, <"gpu", 0>, stable>
    ```
  }];

  let arguments = (ins
    MGP_StableAddressType:$addr,
    MGP_TensorSpecType:$spec
  );
  let results = (outs MGP_DeviceTensorType:$result);
  let assemblyFormat = [{
    $addr `,` $spec attr-dict `:` qualified(type($addr))
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveYieldOp
//===----------------------------------------------------------------------===//

def DeviceGraphCollectiveYieldOp
    : MGP_DeviceGraphOp<"collective.yield", [
  HasParent<"DeviceGraphCollectiveOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for a `mgp.device_graph.collective` device region.";
  let description = [{
    Yields the values this region contributes to the parent op's `$outputs`.
    The parent op appends operands across regions in region order.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat =
    "attr-dict ($operands^ `:` type($operands))?";

  // No-operand builder for the parent's `SingleBlockImplicitTerminator` trait
  // so the auto-generated terminator builder has the right signature; operands
  // are populated after insertion.
  let builders = [OpBuilder<(ins)>];
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveOp
//===----------------------------------------------------------------------===//

def DeviceGraphCollectiveOp : MGP_DeviceGraphOp<"collective", [
  SingleBlockImplicitTerminator<"DeviceGraphCollectiveYieldOp">,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
]> {
  let summary =
    "Bundles a set of per-device device-graph bodies that must launch together.";
  let description = [{
    Holds one region per device. The op is the allocation point for the shared
    stable addresses used to coordinate the device graphs.

    The `$devices` attribute carries one `M_DeviceRefAttr` per region in
    region order, recording which device each region's body represents.

    Each region's body records a single device-graph on its respective device.
    Each region has one entry-block argument, the `!mgp.device_graph.builder`
    the body records into, and terminates with
    `mgp.device_graph.collective.yield`, whose operands append to the parent
    op's `$outputs` in region order.

    The stable address slots the regions exchange values through are not
    values in the IR. `$stableTypes` declares them, one `!mgp.stable_addr<T>`
    per slot, where `T`'s device is the device whose memory holds the slot;
    `mgp.device_graph.stable.get` and `mgp.device_graph.stable.place` name a
    slot by its index into this list and reach it through the builder, which
    carries the collective's slot table at runtime.

    Example:
    ```mlir
    %0:2 = mgp.device_graph.collective builder(%b) stable(
      !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
      !mgp.stable_addr<<4xf32, <"gpu", 1>>>
    ) <"gpu", 0> {
      %p0 = mgp.device_graph.stable.place %b[0], %val : !mgp.tensor<4xf32, <"gpu", 0>>
      mgp.device_graph.collective.yield %val : !mgp.tensor<4xf32>
    }, <"gpu", 1> {
      %a0 = mgp.device_graph.stable.get %b[0] : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
      mgp.device_graph.collective.yield %val : !mgp.tensor<4xf32>
    } : !mgp.tensor<4xf32>, !mgp.tensor<4xf32>
    ```
  }];

  let arguments = (ins
    M_DeviceRefsAttr:$devices,
    MGP_StableAddressArrayAttr:$stableTypes,
    // As mogg.routine.async ops are inlined away, we need to
    // preserve the constant resource attributes.
    DefaultValuedAttr<ArrayAttr, "{}">:$constantResources
  );

  let results = (outs
    Variadic<AnyType>:$outputs
  );

  let regions = (region
    VariadicRegion<SizedRegion<1>>:$graphs
  );

  let extraClassDeclaration = [{
    /// A structured interface to each device graph body.
    struct CollectiveRegion {
      Block *bodyBlock;
      BlockArgument builder;
      DeviceRefAttr device;

      template <typename T=void>
      auto getOps() { return bodyBlock->getOps<T>(); }
    };

    auto graphs() {
      return llvm::map_range(getGraphs(), [this](Region &region) {
        CollectiveRegion result;

        Block *block = &region.front();
        result.bodyBlock = block;
        result.builder = block->getArgument(0);
        result.device = getDevices()[region.getRegionNumber()];
        return result;
      });
    }

    /// Returns the number of regions.
    unsigned getNumRegions() { return getGraphs().size(); }

    /// Returns the block of region `idx`.
    Block &getRegionBlock(unsigned idx) {
      return getGraphs()[idx].front();
    }

    /// Returns the builder block argument for region `idx`.
    BlockArgument getRegionBuilder(unsigned idx) {
      return getRegionBlock(idx).getArgument(0);
    }


    auto getGraphBlocks() {
      return llvm::map_range(getGraphs(), [](Region &region) {
        return &region.front();
      });
    }
  }];

  let builders = [
    OpBuilder<(ins
      "llvm::ArrayRef<M::DeviceRefAttr>":$devices,
      "llvm::ArrayRef<StableAddressType>":$stableTypes,
      CArg<"llvm::function_ref<void(::mlir::OpBuilder &, M::DeviceRefAttr, BlockArgument)>", "nullptr">:$bodyBuilder
    )>
  ];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveCreateOp
//===----------------------------------------------------------------------===//

def DeviceGraphCollectiveCreateOp : MGP_DeviceGraphOp<"collective.create", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"DeviceGraphYieldOp">,
]> {
  let summary = "Builds one device graph per device from a collective body.";
  let description = [{
    Returns a variadic set of `!mgp.device_graph` values, one per region,
    by running each region's build body against its own recorder.

    The op is the multi-device counterpart of `mgp.device_graph.create`:
    each region records a single device graph belonging to a `!mgp.device_context`
    from the `deviceContexts` operand list. The regions share the same
    `theContext` and the same optional `cache`. The `inputs` operand list is
    the flat concatenation of each region's graph inputs, in region order;
    the i-th region's block input arguments consume the next slice of that
    list.

    Each region's block arguments are, in order: the recorder, the model context,
    then the region's slice of the `inputs` operands. The region must terminate
    with `mgp.device_graph.yield`; the yielded value types define the output
    signature of that region's graph.

    The printed form declares each region's entry-block arguments in
    `graph(...)`, `builder(...)` and `inputs(...)` prefix clauses. The shared
    `theContext` follows the op name, then the optional `cache`.

    Example:
    ```mlir
      %g0, %g1 = mgp.device_graph.collective.create %ctx
        graph(%d0: !mgp.device_context<<"gpu", 0>>) builder(%b0)
            inputs(%i0: !mgp.buffer<<"gpu", 0>>) {
          mgp.device_graph.yield %i0 : !mgp.buffer<<"gpu", 0>>
      }, graph(%d1: !mgp.device_context<<"gpu", 1>>) builder(%b1)
            inputs(%i1: !mgp.buffer<<"gpu", 1>>) {
          mgp.device_graph.yield %i1 : !mgp.buffer<<"gpu", 1>>
      } : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
          !mgp.device_graph<(!mgp.buffer<<"gpu", 1>>) -> (!mgp.buffer<<"gpu", 1>>)>
    ```
  }];

  let arguments = (ins
    MGP_ContextType:$theContext,
    Variadic<MGP_DeviceContextType>:$deviceContexts,
    Optional<MGP_DeviceGraphCacheType>:$cache,
    Variadic<AnyType>:$inputs,
    // The stable address slots the regions index into, copied from the
    // `mgp.device_graph.collective` this op was finalized from. Emission
    // allocates one slot per entry on the device its type names.
    OptionalAttr<MGP_StableAddressArrayAttr>:$stableTypes,
    // Positions in `$inputs` of tensors a region reads only for their shape,
    // such as a peer device's tensor whose dimensions size this device's
    // work.
    OptionalAttr<DenseI64ArrayAttr>:$shapeOnlyInputs
  );

  let results = (outs Variadic<MGP_DeviceGraphType>:$outGraphs);
  let regions = (region VariadicRegion<SizedRegion<1>>:$graphs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
      "Value":$theContext,
      "llvm::ArrayRef<mlir::Value>":$deviceContexts,
      "mlir::Value":$cache,
      "llvm::ArrayRef<mlir::Value>":$inputs,
      "llvm::ArrayRef<unsigned>":$inputCounts,
      CArg<"llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location, unsigned, ::mlir::Value, ::mlir::Value, ::mlir::ValueRange)>", "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    // Queryable accessor for the shape only operands to the device graph.
    llvm::SmallBitVector shapeOnlyInputs();
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveExecuteOp
//===----------------------------------------------------------------------===//

def DeviceGraphCollectiveExecuteOp : MGP_DeviceGraphOp<"collective.execute", [
  AttrSizedOperandSegments,
]> {
  let summary = "Replays a set of device graphs built by collective.create.";
  let description = [{
    Triggers replay of the `!mgp.device_graph` values produced by the
    matching `mgp.device_graph.collective.create`.

    The `graphs` operands are the per-device graphs, in the same order as the
    regions of the creating op. The `inputs` operands are the union of the
    live inputs for all graphs, laid out flat as
    `[graph0_inputs..., graph1_inputs..., ...]`; each graph knows how many
    inputs it consumes, so the runtime primitive needs no segmentation.

    The op produces a fresh `!mef.chain` for host-side ordering and a variadic
    set of value outputs, one per graph-output in graph order.
  }];

  let arguments = (ins
    Variadic<MGP_DeviceGraphType>:$graphs,
    Variadic<AnyType>:$inputs
  );
  // The `$chain` is the fresh host-ordering chain; the value `$outputs` mirror
  // the replayed graphs' signature results, which are chain-free buffers and
  // tensors (see `FinalizeDeviceGraphsPass` and `DeviceGraphYieldOp`).
  let results = (outs
    ChainType:$chain,
    Variadic<AnyTypeOf<[MGP_DeviceBufferType, MGP_DeviceTensorType]>>:$outputs);

  let assemblyFormat = [{
    $graphs `inputs` `(` ($inputs^ `:` qualified(type($inputs)))? `)` attr-dict
    `:` type($graphs) (`->` type($outputs)^)?
  }];
  let hasVerifier = 1;
}

#endif // GENERICML_MGPDIALECT_MGPDEVICEGRAPHOPS_TD
