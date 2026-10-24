# Modular graph compiler pipeline description

## Introduction

This document focuses on describing the current state of the optimizations,
lowerings, and analyses in the graph compiler. It describes the system as-is,
using only the source code and a bit of git history without any prior knowledge
of the system.

### Format

The description of a pass should follow the schema:

```text
#### Description
A brief state of purpose and a high-level overview of the pass and how it works.

#### Dialects
Produced and consumed dialects or dependencies.

#### Performance
Notes on performance effects in a broad sense.

#### Additional notes
Optional. Any relevant information in the context of the document.
```

## Dialects

Dialects have extensive documentation. Here's a quick reference:

- **RMO (Relaxed Modular Operations)** - accepts incomplete type information
  from frontends and lowers to MO. Definition:
  `GraphCompiler/include/GraphCompiler/RMODialect/RMODialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (RMO section).
- **MO (Modular Operations)** - framework-agnostic, strongly-typed graph ops
  with value semantics. Definition:
  `GraphCompiler/include/GraphCompiler/MODialect/MODialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (MO section).
- **MOSH (Modular Shape)** - shape/parameter computation for MO graphs.
  Definition:
  `GraphCompiler/include/GraphCompiler/MOSHDialect/MOSHDialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (MOSH mention).
- **MAP (Modular Access Primitives)** - fusion-focused IR that expresses
  executable loop nests/kernels that load, store, and compute over tensors. It
  is the stage where advanced fusion runs (elementwise chains, view, and
  prologue/epilogue fusion around structured kernels). Definition:
  `GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPDialect.td`.
- **MOGG (Modular Operation Generator Graph)** - kernel selection and
  codegen-specific IR, produced from MAP after fusion. Definition:
  `GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGDialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (MOGG section).
- **MGP (Modular Glue Primitives)** - runtime glue dialect with buffer
  semantics; directly translatable to MEF. Definition:
  `GraphCompiler/include/GraphCompiler/MGPDialect/Dialect/MGPDialect.td`.
  Docs: `GraphCompiler/docs/GraphCompiler.md` (MGP section).

Other referenced dialects, not directly part of the compiler:

- **KGEN** - kernel-generation dialect used during JIT compilation. Definitions:
  `Mojo/include/Mojo/KGENDialect/KGENDialect.td` and related KGEN dialect files.
  Docs: `Mojo/docs/compiler/manual/PassesAndIR.md`.
- **HLCF** - high-level control flow dialect used in kernel-generation
  pipelines. Definition: `Mojo/include/Mojo/HLCFDialect/HLCFDialect.td`. Docs:
  `Mojo/docs/compiler/manual/PassesAndIR.md`.
- **LIT** - Mojo frontend dialect used for parsed Mojo IR in kernel compilation
  pipelines. Definition: `Mojo/include/Mojo/LITDialect/LITDialect.td`. Docs:
  `Mojo/docs/compiler/manual/PassesAndIR.md`.
- **POP** - Mojo low-level/runtime support dialect used in kernel generation and
  lowering. Definition: `Mojo/include/Mojo/POPDialect/POPDialect.td`. Docs:
  `Mojo/docs/compiler/manual/PassesAndIR.md`.
- **GRT (GraphRT)** - runtime primitive dialect for execution in GraphRT.
  Definition: `MLRT/include/MLRT/GraphRT/GRTDialect/GRTDialect.td`. Docs:
  `MLRT/docs/GraphRT/KeyConceptsInGraphRT.md`.
- **MEF** - low-level execution format dialect used by the MEF executor.
  Definition: `MLRT/include/MLRT/GraphRT/MEFDialect/MEFDialect.td`. Docs:
  `MLRT/docs/GraphRT/ModularExecutorFormat.md`.

## Analyses

### MojoLibraryAnalysis

#### Description

Loads and caches Mojo kernel libraries (standard + extra paths) and exposes
lookup helpers for registered kernel generators and symbols. It materializes a
module of exported generators and serves queries needed by kernel selection and
lowering.

#### Dialects

Consumes `mo` and depends on `lit` support for loading Mojo packages and
generators.

#### Performance

Amortizes library parsing/validation across passes by caching the loaded
libraries; reduces repeated kernel discovery costs during lowering/codegen.

#### Additional notes

Initialized with `kernelLibraryPaths`, then lazily loads Mojo packages and
builds a generator table used by lowering passes (e.g., MO to MOGG).

Custom kernels are registered on Mojo side through
`@extensibility.register(...)`. The compiler will later use the symbol to match
the kernel during lowering. Custom ops can target these using
`mo.custom [symbol = "MyOp"]` (see <ExtensibilityUserGuide.md>).

The analysis parses Mojo packages/bytecode into MLIR, walks every
function/struct, and builds multiple lookup tables with registered kernels,
execute/shape/view functions, opaque-type map. By default, the main two
components are loaded:

- MOGGKernelAPI - a registry of hand‑fused/high‑level kernels and helpers.
  Normally, accessed via `mo.custom` and targeted in the PatternFusion pass.
- MOGGPrimitives - baseline Mojo kernels/primitives used by the graph compiler
  as standard kernel implementations (e.g., core ops like tensor construction
  `mgp.tensor.create` or allocation `mgp.buffer.alloc`, views, elementwise
  helpers, etc.). These are normally introduced in the MO->MGP lowering
  (MoToPrimitives and related lowering passes) and then included through kernel
  stubs produced by EmitMojo pass.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h>
Implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.cpp>

### ParametricTopologyAnalysis

#### Description

Extends SSA dataflow with parametric dependencies (shape/parameter attributes)
and nested-region providers, enabling correct dependency queries and
topological sorting when attributes (not SSA values) drive ordering.

#### Dialects

Operates on `mo` graphs and their nested regions; models attribute-level
dependencies (e.g., `kgen` parameter attrs) alongside SSA.

#### Additional notes

This is a custom analysis (technically, not an MLIR analysis) that allows passes
to reason about true dependency order that are not expressed via SSA values.
Provides `users()`/`providers()` that union SSA and parametric dependencies, and
`topologicallySortAllOpsWithin`/`computeTopologicalSorting` helpers. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h>

### BufferLifetimeAnalysis

#### Description

Builds dependency trees over `mgp.model` blocks to reason about execution order
and buffer lifetimes, enabling safe memory planning and buffer reuse.

#### Dialects

Operates on `mgp` (ModelOp) and buffer operations.

#### Performance

Enables downstream memory planning passes to reuse buffers by providing fast
`executesAfter`/LCA queries over dependency trees.

#### Additional notes

Tracks a per-block `DependencyTree` with pre/post indices, supports dominance
and LCA queries, and includes alias-aware alloc-use discovery via
`getAllocUses`. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.h>

## Lowering passes

The path from `mo` to `mogg` runs through the MAP dialect, where advanced
fusion happens: `MOToMAP` lowers each eligible `mo` op to a single `map`
kernel, the MAP pipeline fuses and prepares those kernels, and `MAPToMOGG`
crosses into the codegen-focused `mogg` IR. The passes below are listed in
pipeline order (`buildMapPipeline` and `buildMoggPipeline` in
`GraphCompiler/lib/GraphCompiler.cpp`). There is no longer an `MOToMOGG`
conversion; the legacy MOGG fusion passes (which fused by embedding producers
in a root kernel's output region) have been removed in favor of this flow.

### MOToMAP

#### Description

Lowers known `mo` dialect ops into one `map` kernel each. It is the entry point
of the advanced-fusion pipeline.

#### Dialects

Consumes `mo` and produces `map`, reusing the same `mogg.kernel.decl`
machinery as the rest of kernel selection (`MojoLibraryAnalysis`).

#### Additional notes

Kernel selection works the same way the old `MOToMOGG` lowering did: the op is
mapped to a registered Mojo kernel name/function label (custom ops use their
symbol directly), the kernel is resolved through `MojoLibraryAnalysis`, and the
op is replaced with a `map` kernel. Ops with no compatible kernel are left for
later lowering with a graceful diagnostic. Control-flow ops and runtime
plumbing ops (buffer transfers, materialization, views, list ops, and similar)
are preserved and survive until `MOGGToMGP`.

Source: `GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp`.
Pass definition:
`GraphCompiler/include/GraphCompiler/Conversion/MOToMAP/MOToMAP.td`.

### FuseKernels (advanced fusion)

#### Description

Runs the advanced-fusion patterns over the MAP dialect, reducing the number of
`map` kernels by inlining producers into consumers.

#### Dialects

Consumes and produces `map` (anchored on `mo.graph`), depends on `mogg`.

#### Performance

Fewer, larger kernels means fewer dispatches and fewer materialized
intermediates. Fusion eligibility is a function of kernel structure (the Mojo
types a kernel uses for its inputs/outputs and the fusion interfaces it
implements), not of op identity.

#### Additional notes

The pass drives the fuser classes in `GraphCompiler/lib/MAPDialect/Transforms`:
`EpilogueFuser`, `ElementwiseFuser`, `ViewFuser`, and `PrologueFuser` (plus
`ReshapeEpilogueFuser`). It repeats epilogue, elementwise, view, then prologue
fusion until a round fuses nothing, since folding one link of a chain can
expose the next. It reports the number of kernels fused as a pass statistic.
The per-fuser `map-fuse-*` passes (`map-fuse-epilogue`, `map-fuse-elementwise`,
`map-fuse-views`, `map-fuse-prologue`, and the generic `map-fuse-input` /
`map-fuse-output`) exist only to let lit tests exercise one fuser in isolation;
they are not part of `buildMapPipeline`.

Source: `GraphCompiler/lib/MAPDialect/Transforms/`.
Pass definition:
`GraphCompiler/include/GraphCompiler/MAPDialect/Transforms/MAPTransforms.td`.

### FuseMutableLoads

#### Description

"Fuses" an `mo.mutable.load` into a `map.iter.elementwise` so the loop reads the
buffer-semantic tensor directly. Enabled by the `fuseMutableLoadStore` option.

#### Dialects

Consumes and produces `map` (on `mo.graph`).

#### Additional notes

Replaces the value-semantic tensor from the load with the buffer-semantic one,
threads the load's input chain into the `map.iter.elementwise`, and adds the
load's output chain to the loop's outputs.

Pass definition:
`GraphCompiler/include/GraphCompiler/MAPDialect/Transforms/MAPTransforms.td`.

### AllocateAccumulatingBuffers

#### Description

After fusion settles, gives every fused-output epilogue a real store pointer so
`MAPToMOGG` never hands a lowered epilogue a dangling base pointer.

#### Dialects

Consumes and produces `map`, depends on `mo`.

#### Additional notes

Normally the pointer source is the tensor the epilogue stores into. For a
mixed-precision accumulating kernel (a matmul/conv/concat whose epilogue changes
dtype), a fresh accumulation buffer in the pre-cast dtype is allocated as an
extra kernel output and used as the pointer source, while the store target stays
the real (post-cast) output. Non-accumulating kernels get no extra buffer.

Pass definition:
`GraphCompiler/include/GraphCompiler/MAPDialect/Transforms/MAPTransforms.td`.

### RaiseBackContiguousViews

#### Description

Raises `map.iter.elementwise` view kernels that never got fused back into ops
the rest of the pipeline can lower without a copy.

#### Dialects

Consumes `map` and produces `mo`/`mogg`.

#### Additional notes

Runs after `FuseKernels` reaches a fixed point and before
`OutlineStructuredKernelBodies`. A contiguous slice becomes `mogg.slice` (a
zero-copy buffer slice, later lowered by `MOGGToMGP` to `mgp.tensor.slice`),
and a reshape becomes `mo.static.reshape` (a non-copying primitive). Only slice
and reshape are raised back, not broadcast or transpose.

Pass definition:
`GraphCompiler/include/GraphCompiler/MAPDialect/Transforms/MAPTransforms.td`.

### OutlineStructuredKernelBodies

#### Description

Wraps each structured MAP loop in a `map.iter.opaque` region so sequential setup
can live outside the structured loop body.

#### Dialects

Consumes and produces `map`.

#### Additional notes

The loop's shape and body move into a nested `map.iter.elementwise.body`, which
reaches the wrapper's inputs/outputs by SSA dominance. After outlining, nothing
inspects the inside of the `map.iter.opaque` again. `LoopInvariantViewMotion`
(below) then runs over these regions.

Pass definition:
`GraphCompiler/include/GraphCompiler/MAPDialect/Transforms/MAPTransforms.td`.

### LoopInvariantViewMotion

#### Description

Hoists chains of index-transform ops out of a structured loop body into the
enclosing `map.iter.opaque`, so each transform runs once per kernel invocation
rather than once per index.

#### Dialects

Consumes and produces `map` (per `mo.graph`).

#### Additional notes

This rewrite is mandatory, not an optimization: an unhoisted index-transform op
has no `LowerableToMOGGInterface` and would strand `map-to-mogg`. For each
`map.load`, it walks backward from the index operand through any chain of
index-transform ops (today `map.index.slice`) and hoists them into "stride" ops
(today `map.index.slice.stride`) in the enclosing opaque region. The pass
enforces that no index-transform op survives inside a
`map.iter.elementwise.body`.

Pass definition:
`GraphCompiler/include/GraphCompiler/MAPDialect/Transforms/MAPTransforms.td`.

### MAPToMOGG

#### Description

Lowers `map` dialect kernels into `mogg` codegen ops: dynamic dims become SSA
values, contexts are added, and tensors are built from pointers. This is where
the fusion-focused `map` IR crosses into the code-generation-focused `mogg` IR.

#### Dialects

Consumes `map` and produces `mogg` (depends on `mo` and `mosh`).

#### Additional notes

Each lowered kernel becomes a `mogg.routine` container whose body calls the
selected kernel (`mogg.call.opaque` for opaque kernels, `mogg.call.elementwise`
/ `mogg.call.foreach` for elementwise loops). The `enableComputeLambdas` option
controls whether an epilogue feeding a `_FusedComputeOutputTensor` argument gets
a compute lambda or a classic store lambda.

Source: `GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp`.
Pass definition:
`GraphCompiler/include/GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.td`.

### ConformCallOperands

#### Description

Conforms each `mogg` call op's operands to the callee's signature by inserting
the operand-conversion ops the callee requires.

#### Dialects

Consumes and produces `mogg`.

#### Additional notes

A scalar argument's tensor is wrapped in `mogg.scalar.create`, a variadic run is
packed into a `mogg.tuple.create`, and a trait-bound argument (`Tensor` /
`TileTensorable`) is wrapped in a `mogg.trait.create` so the raw
`!mogg.tensor` reaches the kernel as the conforming trait argument. This is
deferred out of `MAPToMOGG` to keep the fusion IR simpler and each emission a
local, 1:1 lowering.

Pass definition:
`GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGPasses.td`.

### OutlineRoutines

#### Description

Outlines each `mogg.routine` body into a module-level `mogg.func`.

#### Dialects

Consumes and produces `mogg` (and `mo`).

#### Additional notes

Moves each `mogg.routine` body into a `mogg.func` labeled with the MAP ABI
and replaces the body with a single `mogg.call` forwarding the kernel's
pointer/dim/context block arguments to the stub. The `mogg.routine` container is
kept because it carries the `!mo.tensor` interface that `MOGGToMGP` later
lowers. The MAP ABI label lets `EmitMojo` branch on the advanced-fusion
emission path.
`WrapFuncsWithTrace` then wraps each stub body in a per-kernel trace range.

Pass definition:
`GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGPasses.td`.

### MOGGToMGP

#### Description

Converts `mo`/`mogg` ops into `mgp`/`mef` primitives at the module level,
producing a runtime-oriented graph that can be further lowered and optimized.

#### Dialects

Consumes `mo`/`mogg` and produces `mgp`/`mef`, with runtime support from `grt`,
`pop`, and `index`.

#### Performance

Enables later MGP pipeline optimizations and kernel outlining by moving from MO
to buffer/primitive semantics. Supports module-level orchestraction
(topological dependency order).

#### Additional notes

Runs as a module pass because it lowers across all graphs at once and needs to
avoid per-graph setup overhead. The pass lowers `mo.graph` into an `mgp.model`
with `init` and `execute` regions.

The pass collects all `mo.graph`s and builds a dependency map from `mo.call`.
Then lowers graphs in the topologically sorted order while detecting dependency
cycles. Each graph is lowered to a single `mgp` model, walking its body,
arguments and patching the control flow with `grt`. The lowering of ops goes
either through the `LowerableToMGPInterface` or dispatches to a specialized
handler if an op has one. The advanced-fusion `mogg.routine` container (produced
by `MAPToMOGG` / `OutlineRoutines`) implements
`LowerableToMGPInterface` directly and is lowered through that path. `mo.call`s
are lowered to `mgp.call`s
with implicit operands that include chains, context, weight registry, device
contexts.

Within blocks, operations are sorted in levelwise topological order. This is a
workaround for `mo.if` preventing weight transformations from moving to init
region (`mgp.buffer.alloc` for weights is chained after `grt.if`, preventing
ExecInvariantCodeMotion pass from moving the weight transformation to model
init). Early operations reliably sit before control-flow.

At this stage, bufferization (value-semantic to buffer-semantic transition) is
performed. The logic lives in `MOOpLoweringState`. For inputs, the lowering
extracts the underlying buffer and spec for each tensor from an already-lowered
`mgp.tensor` value. Outputs are lowered in destination-passing style: create a
tensor spec, compute size, allocate a buffer, and then build an
`mgp.tensor.create`.

The difference between the shape of a tensor being static or dynamic is in the
materialization of the tensor spec. For dynamic, runtime dims are expressed via
`mosh`. Constants may introduce additional buffer allocation + transfer.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

## MGP passes

### LowerPrimitives

#### Description

Wraps eligible MGP/POP/index ops into mogg.routine.async kernels and
applies the MOGG ABI to prepare for fusion and Mojo emission.

#### Dialects

Consumes and produces `mgp` and depends on `kgen` (`lit`, `pop`) plus `mef`,
`mogg`, `index`.

#### Additional notes

The pass aids the sequence fusion pass by lowering some top-level model ops to
uniform compiled region kernels. It walks the ops that implement
`EmitsMojoInterface`, wraps them into a new `mogg.routine.async`, and
rewrites the complied region's ABI to MOGG ABI. The MOGG ABI means the
async‑value calling convention used for mogg.routine.async: inputs
are unpacked from async wrappers on entry, and outputs are packed back into
async wrappers on exit, with special handling for chains and borrowed buffers.
The ABI lowering does three things:

1. For arguments marked "shouldUnpack" (by emits mojo interface), it inserts
   `mogg.async.unpack` (or borrow‑aware variants) and rewrites uses to the
   unpacked values.
2. For results marked "shouldPack", it wraps them with `mogg.async.pack` /
   `mogg.async.pack.borrow` variants. Chains are replaced by
   `mogg.async.ready` and sunk with `mogg.chain.sink` to preserve side effects.
3. Any chain arguments are replaced with a dummy chain, so the body doesn’t
   accidentally use real chain inputs.

The pass also performs special lowering for `index.constant` ->
`mgp.buffer.constant`, `mgp.buffer.to_index` and a resource attribute (no kernel
hashing on constants).

Relies on `MojoLibraryAnalysis` but never uses it.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/Conversion/MOGGToMGP/LowerPrimitives.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### LookupConstantResources

#### Description

Replaces `mgp.buffer.constant` uses inside `mogg.func` with
`mgp.resource.constant` indices that reference the stub’s constant resource
table. This makes constants comparable by index rather than by value.

#### Dialects

Consumes and produces `mgp` (within `mogg.func`).

#### Performance

Improves kernel stub merging and downstream compilation by avoiding large
constant payloads in the IR and allowing identical kernels to share constants.

#### Additional notes

The pass runs late in the pipeline, right before emitting Mojo. It walks
`BufferConstantOp` in a `FuncOp`, resolves each constant against the
stub’s resource list, then replaces it with `ResourceConstantOp`, and resets
the resource array to a dummy list of the same size.

The pass expects all the resources to be found in the constant resources array
attached to the `FuncOp` by the `MO::ConstantOp` lowering. The resources
are referenced by index, so that multiple kernels can access the same resource.
This has many obvious benefits including memory efficiency, avoiding hashing
kernels by constant values, and compilation efficiency. Earlier in the
pipeline, index constants are converted to `mgp.buffer.constant` (by the
LowerPrimitives pass) with a resource attribute for the same reason.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/LookupConstantResources.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### ExecInvariantCodeMotionPass (EICM)

#### Description

Moves computations from the `execute` block to the `init` block when they do
not depend on `execute` arguments. Uses `!mgp.context` slots to forward results,
with heuristics to avoid lifting trivial or awkward-to-cache ops.

#### Dialects

Consumes and produces `mgp` (ModelOp), with control-flow support from `grt` and
buffer plumbing from `mef`.

#### Performance

Reduces per-execution overhead by hoisting invariant work to initialization and
enables reuse of cached buffers.

#### Additional notes

The goal is to hoist execute‑invariant work to init, so it is computed once per
model rather than on every execution, while respecting side‑effect and
control‑flow safety. The pass runs early in the MGP pipeline, after MO has been
lowered to `mgp.model`(only available after MoToPrimitive) and after some basic
cleanup, but before memory planning and other buffer‑lifecycle optimizations
(these assume the init/execute split is already settled).

The pass defines "liftable" ops (buffer‑creating ops and chain‑creating ops) and
recomputable ops (pure ops) that can be moved or duplicated safely. The
condition is that an op does not:

- depend on execute‑block arguments,
- have side effects,
- is inside control‑flow,
- is explicitly unliftable (e.g., external constants that can be mutated
  outside the system; this can be a pytorch tensor).

The condition is determined by a set of analyses. A dense forward relation
analysis over values/operands propagates an actual dependency on the argument
or if an value is produced by an unliftable op, including control flow that is
considered intrinsically unliftable. A set of backward analyses make sure that:

- Ops whose results are mutated by unliftable ops are not lifted (values cached
  in `!mgp.context` are read‑only).
- Liftable ops whose results are required by unliftable ops which can not be
  conveyed via `!mgp.context` are cloned.
- Trivial ops with unliftable consumers are not lifted (recomputing is cheaper
  than caching).

The pass runs the analyses to convergence. Then the hoisting is performed. The
latter must consider preserving side-effect ordering (this involves some
gating placeholder chain creation until the pass can compute the real
dependency chain).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### FixupOverhangsPass

#### Description

Finds chain results with no uses and adds waits on the block terminator so all
chains are awaited before graph exit.

#### Dialects

Consumes and produces `mgp` (ModelOp).

#### Performance

Adds minimal extra dependencies to preserve correctness and ensure async errors
are not silently dropped.

#### Additional notes

Part of the JIT compilation pipeline. Walks each block in `mgp.model`, collects
unused results (one per op), and uses `alsoWaitForChains` to add dependencies on
the block output. An unused result is assumed to be a dangling chain (this is an
IR invariant at this stage). The pass fulfills the assumption of the model being
'overhang free' (see
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/AsyncDataflowDependencies.h>)

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/FixupOverhangsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### SimplifyExtendLifetimes

#### Description

Fuses `mgp.extend.lifetime` into a producer `mogg.routine.async` by
threading the extend-lifetime inputs directly into the kernel op and removing
the explicit lifetime extension op.

#### Dialects

Consumes `mgp`, rewriting uses of `mogg.routine.async` and `mef`
chain values.

#### Performance

Reduces IR size and avoids standalone lifetime-extension ops by folding them
into the kernel boundary.

#### Additional notes

`mgp.extend.lifetime` are created at MO to MGP lowering (during MOGGToMGP)
whenever a transfer is introduced (e.g., host->device or device->host copies;
so, both source and destination buffers). When buffer builder emits a transfer,
it adds an `mgp.sync` on the copy chain and then a `mgp.extend.lifetime` to keep
the relevant buffer alive until that sync completes.

The pass is run once the kernel boundaries and operand lists are settled, so
after the fusion, right before kernel outlining (after which there's not enough
information remaining).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/SimplifyExtendLifetimesPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### ConcatInPlace

#### Description

Rewrites `mgp.buffer.concat` into a single output allocation plus
`mgp.buffer.split.borrow`-style slices so that producers write directly into
the final output buffer.

#### Dialects

Consumes and produces `mgp` (ModelOp) with chains and `index` buffer-size values
used to compute offsets.

#### Performance

Eliminates explicit concat copies by turning concatenation into in-place writes
to slices of the output buffer.

#### Additional notes

Replace an explicit concat that writes into a freshly allocated output buffer
with a split‑borrow pattern: allocate one big buffer, slice it into per‑input
buffers, and redirect all writers to write directly into those slices. This
makes concat "in‑place" and removes the op itself.

Runs over the execute region, finds `mgp.buffer.concat` ops whose input buffers
and output buffer are all explicitly allocated and distinct, then computes
offsets from the runtime sizes. It inserts slices of the output buffer at the
input-buffer allocation sites, replaces each input buffer with its slice, and
erases the original allocs and concat op.

The pass assumes the concat input chain is only ready once all input buffers
are ready, so it replaces the concat output chain with the input chain.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/ConcatInPlacePass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### OutlineAsyncRoutines

#### Description

Outlines `mogg.routine.async` bodies into `mogg.func` ops and
replaces the original region with `mogg.execute.compiled` calls.

#### Dialects

Consumes and produces `mogg`/`mgp` within a `ModuleOp`.

#### Performance

Outlining "stabilizes" the kernel boundaries before the Mojo codegen making
them explicit and deterministic. The entry points get uniform and addressable
(by symbol reference) which enables dedup/merge of identical stubs downstream.

#### Additional notes

Each `mogg.routine.async` is outlined into a named mogg.func
(`__<stubs it calls>_async`, or `__primitive_<n>_async` when it calls none),
and the original site becomes a `mogg.execute.compiled` call.
Sets the device from the original op, and forwards summary/async/constant
resource attributes. Uses `ABIKind::MGP` (void-style; multiple results via
opaque buffers) for the stub.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/OutlineAsyncRoutines.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### LiftBufferCacheSlots

#### Description

Lifts inline `index.constant` ops that feed a `mgp.buffer.get_cached` slot
operand out of kernel bodies and into kernel-stub function arguments.

#### Dialects

Consumes and produces `mogg` (within kernel stubs).

#### Additional notes

Two kernels that differ only in a cached-buffer slot index get structurally
identical bodies after this runs, so `MergeIdenticalFuncs` can then
collapse them. Runs in `buildMgpPipeline` before the merge (it was previously
nested inside `EmitMojo`).

Pass definition:
`GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGPasses.td`.

### MergeIdenticalFuncs

#### Description

Merges structurally identical `mogg.func` ops using their hashes, so the
Mojo compiler elaborates each distinct kernel once.

#### Dialects

Consumes and produces `mogg` (`ModuleOp`).

#### Performance

Collapses near-duplicate kernels (for example per-device copies across EP
shards), cutting kernel count and Mojo compile time.

#### Additional notes

`CanonicalizeFuncsForMerge` (erases device ids and symbolic dim names inside
stubs) and `LiftBufferCacheSlots` run first so otherwise-identical stubs hash
equal.
`SymbolDCE` later removes the stubs this pass leaves without callers. These
passes run in `buildMgpPipeline` just before `EmitMojo`; they were previously
nested inside it.

Pass definition:
`GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGPasses.td`.

### AddStackTracesToAssertsPass

#### Description

Augments `mgp.assert` messages with build-time Python stack traces derived from
operation locations.

#### Dialects

Consumes and produces `mgp` (ModelOp); uses `rmo` location metadata.

#### Additional notes

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/AttachPythonStackTraces.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### VerifyModelArguments

#### Description

Inserts runtime shape checks for `mgp.model` execute arguments to ensure
provided tensors match statically known dimensions.

#### Dialects

Consumes and produces `mgp` (ModelOp), using `index` comparisons and `pop`
boolean ops for predicates.

#### Performance

Adds runtime checks on model entry. Overhead is proportional to the number of
static dimensions in the execute arguments.

#### Additional notes

The pass scans execute arguments, skips non-tensors and fully dynamic shapes,
extracts the runtime tensor spec, and builds per-dimension equality checks.
A device tensor should have at least one static dim and a known rank. For those,
the pass combines the checks with `pop.and` and emits an `mgp.assert`.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/VerifyModelArgumentsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### PrepareDevicePrimitives

#### Description

Specializes device-generic `mgp` primitives by renaming ops to include the
target device label and rewriting device-ref attributes into plain device ids.

#### Dialects

Consumes and produces `mgp` (ModelOp), using device metadata (`#M.device_ref`)
from `m`.

#### Performance

Avoids runtime device-label reconstruction and enables GraphRT dispatch to use
device-specific primitive names directly.

#### Additional notes

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/PrepareDevicePrimitivesPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### DeferAllocsPass

#### Description

Defers `mgp.buffer.alloc.*` operations until just before the unique mutating
op that writes the buffer, and wires the allocation to the same input chains as
that mutator.

#### Dialects

Consumes and produces `mgp` (ModelOp).

#### Performance

Reduces allocations live ranges and preserves ordering.

#### Additional notes

The pass walks both execute and init sections. For each alloc (ops implementing
`MGPBufferAllocOpInterface`) it finds the unique mutating op (either a real
mutating op or a `mgp.buffer.slice` (these can come from the dynamic planning
pass that runs just before; the pass started handling slices after the
introduction of the planning pass); only single uses are considered) that writes
to the buffer and moves the allocation immediately before the mutating op. The
mutator's input chains are inserted into the alloc so it can't execute too
early. Moves each alloc before its unique mutator and adds chain dependencies
via `alsoWaitForChains`, asserting that allocs accept input chains. Uses
`ExecuteState` and applies to both `init` and `execute` regions. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/DeferAllocsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### DynamicMemoryPlanning

#### Description

Performs runtime memory planning by grouping eligible buffer allocations and
replacing them with slices of a single pooled buffer.

#### Dialects

Consumes and produces `mgp` (ModelOp), using `mgp.buffer.plan` and
`mgp.buffer.slice`.

#### Performance

Reduces device memory footprint by reusing a pooled allocation while preserving
buffer lifetimes.

#### Additional notes

The goal of the pass is to build a single device-local memory pool (sized) for
eligible device allocations that would replace all individual `mgp.buffer.alloc`
with a slice into that pool, based on a dynamic lifetime plan.

For some context, the direction of the memory planning evolution somewhat
follows this design doc draft:
<https://www.notion.so/modularai/Universal-Static-Memory-Planning-1672fa070a0845deab513e1678a0d6db>
(the idea from the design doc is still compatible with the current
implementation, but it is only a pure mgp rewrite). Aside from additional
optimization opportunities, the fact that device allocation during execution
requires host invocation makes the transformation necessary.
The pass uses `BufferLifetimeAnalysis` to build a dependency tree and compute
pre/post indices for lifetime ranges. With those, it constructs a
`mgp.buffer.plan` op that has an implementation in MGPPrimitives library (a
single-pass greedy algorithm that tries to reuse non-overlapping memory blocks).
The op stores a precomputed NxN sharing matrix (`canShare[i][j] = 1` iff
allocations `i` and `j` can occupy the same slot) rather than raw pre/post
indices, making the reuse structure directly readable and canonical. The op
returns a high watermark value and a set of offsets that are consumed by
`mgp.buffer.slice`s, replacing original allocations. The high watermark value
is computed at runtime and is used for the allocation. The alignment is computed
as the least common multiple of all the allocations to be compatible with all
the slices.

Only non-data-dependent allocations (so the sizes are trivially computable) in
the execute block with uses and without untraceable aliases are considered.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### SimpleBufferReuse

#### Description

Reuses existing `mgp.buffer.alloc` buffers when a later allocation has
compatible size, device, alignment, and strictly ordered uses.

#### Dialects

Consumes and produces `mgp` (ModelOp).

#### Performance

Reduces allocation count and peak memory usage in straightforward reuse cases.

#### Additional notes

This is the first, low‑risk buffer‑reuse pass in the MGP pipeline. It runs after
EICM and before DynamicMemoryPlanning, so it can reduce the number of allocs
(and their lifetimes) that the planner has to consider. The pass reuses one
buffer alloc for another if it can prove all uses of the current buffer execute
after all uses of the candidate buffer. The pass groups allocs by
`(runtimeSize, device, alignment)`, uses `BufferLifetimeAnalysis` dependency
trees to ensure all uses of one buffer execute after the other, and avoids
reuse when allocations share a use. Skips allocs inside
`mogg.routine.async`.

The pass marks BufferLifetimeAnalysis preserved though the IR is modified (so
the analysis is invalid). The expectation is that the next pass (dynamic memory
planning) checks the uses of the allocs, but not allocs themselves.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/SimpleBufferReusePass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### SimplifyChainsPass

#### Description

Removes redundant inputs from `grt.chain.create` by dropping chains that are
already implied ready by other inputs.

#### Dialects

Consumes and produces `mgp`/`grt` (ModelOp).

#### Performance

Reduces chain fan-in and unnecessary dependencies.

#### Additional notes

Uses implied-ready queries over transitive chain dependencies (via a local
backward walk over the operand graph to answer "does chain A imply chain B?";
this is less precise than the former implementation through
`AsyncDataflowDependencies` but gives much better compilation times), updates
chain.create operands in parallel (if threading is enabled), then re-runs chain
canonicalization patterns.

Some limitations of the local analysis:

- no equivalence/aliasing reasoning, pure structural reachability over operands
- no transitive semantic rules beyond operands
- no set simplification via relation algebra
- no "always ready" knowledge for empty merges folding

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/SimplifyChainsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### SequenceFusion

#### Description

Fuses sequences of `mogg.routine.async` ops into larger regions based
on dataflow and fusion-compatibility rules.

#### Dialects

Consumes and produces `mogg`/`mgp` (ModelOp), relying on MOGG sequence-fusion
analyses.

#### Performance

Reduces kernel count and overhead by merging compatible compiled regions.

#### Additional notes

SequenceFusion is the final MGP‑level kernel fusion pass. It fuses sequences of
mogg.routine.async ops into a single compiled region. The pass runs
after LowerPrimitives, so it doesn't work with `mogg.routine`. It reuses
`sourceDestinationSequenceFusionAnalysis` to build a subgraph of fusible ops.
Fusion is done in three phases:

1. Fuse producers that are allowed to be duplicated into multiple consumers.
2. Fuse when the producer is single-use.
3. (optional) Fuse ops that are only transitively related (no
   producer-consumer edges in the graph) as long as no cycles are created and
   no new dependencies between computationaly significant (cannot be fused)
   ops.
The fusion chains are restricted by a maximum of 16 kernels as a compilation
time improvement (see
<https://linear.app/modularml/issue/GEX-3054/epic-mitigate-compilation-time-regression>).

The pass relies on `ParametricTopologyAnalysis` for deterministic ordering
(preserved), and preserves region attributes like async flags, resources, and
fusion capability.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/SequenceFusion.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### EmitMojo

#### Description

Emits Mojo source for `mogg.func` and compiled regions, stores it as a
module resource, and wires models to load the emitted kernels at init time.

#### Dialects

Consumes `mogg`/`mgp` and depends on `mef`/`grt` (from the nested cleanup) and
`kgen`/`lit`/`pop` (from library analysis).

#### Performance

Decouples kernel compilation between GC and MOCO. There's optional caching for
kernels with deduplication logic to limit compile times.

#### Additional notes

EmitMojo is the last step of the MO to MGP pipeline before JIT compilation, and
it now only emits Mojo text. The stub-preparation work it used to do nested
inside itself has been hoisted into `buildMgpPipeline` as its own passes that
run before EmitMojo (see `GraphCompiler/lib/GraphCompiler.cpp`):
`LiftBufferCacheSlots`, `CanonicalizeFuncsForMerge`,
`MergeIdenticalFuncs`, `RaiseBackIndexConstant`, `FixupOverhangsPass`,
`SymbolDCE`, `AddProfileScopeTracing`, and `AddTryExceptBlocks`. By the time
EmitMojo runs, the kernel stubs are already deduplicated and finalized.
EmitMojo inserts `mogg.compiled.load` ops in model init and removes
kernel stubs after emitting the Mojo blob.
The last step builds a `EmitMojoState` (holds contexts, metadata; responsible
for symbol resolution via `MojoLibraryAnalysis`, constant resolution, type
conversion, and lifetime management), imports Mojo stdlib together with custom
kernel packages and walks kernel stubs.
For each `mgp.model`, the pass gathers all kernel symbols references by the
execute section, and adds a `mogg.compiled.load` op into init that points
to the Mojo blob. The op is later lowered to `mgp` buffer load while
JitCompileKernels provides the pointer to the correct resource.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/EmitMojo.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### LowerCompiledRoutines

#### Description

Lowers `mogg.load.compiled.kernel` to `mgp.module.load` and rewrites
`mogg.execute.compiled` into `mgp.execute.generic` using module/kernel indices.

#### Dialects

Consumes `mogg` and produces `mgp`/`grt` within a `ModuleOp`.

#### Performance

Moves compiled-kernel execution into the runtime-friendly MGP form with
explicit module loading and indexing.

#### Additional notes

This is the final lowering step that turns the EmitMojo/JIT artifacts into
runtime MGP ops. It runs immediately after JitCompileKernels. After the pass,
all the symbolic information (e.g., kernel names, blob resources) accessed by
`mogg` ops is finalized into `mgp` runtime ops with index-based access.
Tracks module/kernel indices per load, rewrites executes in both init and
execute blocks (finding the correct `!mgp.context`), replaces the module-load
placeholder chain, removes `mogg.to_mgp_from_opaque` in execute, and fixes
isolated-from-above bindings. Module loads are forced to CPU.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/LowerCompiledRoutines.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### RemoveRedundantOperands

#### Description

Removes redundant inputs/results on `mogg.routine.async` ops after
sequence fusion, collapsing duplicate operands and unused chain/results.

#### Dialects

Consumes and produces `mogg`(ModelOp), depends on `mef` for chain types.

#### Performance

Reduces async value count and kernel signatures by eliminating unused or
duplicate operands/results.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/RemoveRedundantOperands.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.td>

### GlobalizeSlotIndices

#### Description

Globalizes `mgp` context slot indices across all models in a module, ensuring
unique slot ranges for subgraph calls.

#### Dialects

Runs on `mgp` modules (`ModuleOp`), updating `mgp.context.create` and
`mgp.call` ops; `kgen` utilities.

#### Performance

Prevents slot conflicts when models are combined or inlined; minimal overhead.

#### Additional notes

When a module contains multiple mgp.model ops that call each other, each model
may define local slot indices (from its `mgp.context.create`). This pass assigns
global slot offsets so that nested model calls use globally unique slot indices.
Builds an MGP call graph (represents a model op where edges are calls between
models; small wrapper around `KGEN::SCCGraph`, so internally it computes SCCs
and schedules analysis/rewrite in post-order, bottom-up), computes per‑model
slot usage, and inserts per‑call slot offsets (index constants) while marking
calls as globalized.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/GlobalizeSlotIndicesPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

### InlineMGPCalls

#### Description

Inlines `mgp.call` subgraphs into their caller models, cloning init/execute
regions and removing the call operations.

#### Dialects

Runs on `mgp` modules (`ModuleOp`) with `mgp.model` and `mgp.call`.

#### Performance

Eliminates call overhead and enables downstream optimization across model
boundaries; increases IR size.

#### Additional notes

Part of the JIT compilation pipeline, run early after LowerCompiledRoutines.
Builds a call graph, rejects recursion, offsets runtime slot indices in cloned
ops, prefixes placeholder weight names, updates init output chains, and erases
inlined subgraph models when unused. Also updates `mgp.context.create` slot
counts after inlining.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MGPDialect/Transforms/InlineMGPCallsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.td>

## RMO passes

### LegalizeRMOOps

#### Description

Lowers `rmo` ops into `mo` ops, primarily via the `MOAnalogue` interface (maps
`rmo` to `mo` 1 to 1) and rebinds that align operand/result types with their MO
counterparts. It also expands composite `rmo` ops into simpler `rmo`/`mo`
sequences when needed.

#### Dialects

Consumes `rmo` and produces `mo` (GraphOp). Depends on both `rmo` and `mo`
dialects.

#### Performance

Primarily a correctness/legalization step; enables downstream MO pipeline
optimizations by producing strongly typed MO ops early.

#### Additional notes

Implemented as a greedy rewrite pipeline. Includes specialized lowerings for
mutable ops, broadcast/concat/conv/pool/matmul/reshape/select, and rebinds.
`skip-rmo-rebinds` can be used to keep rebinds for debugging.

The pass infers MO types (type inference implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/RMODialect/RMOOps.cpp#L367-L873>,
MO-analogue type mapping:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/RMODialect/RMOMOAnalogueOps.cpp>)
and rebinds MO ops outputs back to the original RMO result type. Ops that don't
have a direct mapping or require some additional logic are handled in two main
ways:

- Specialized legalization patterns, e.g., `rmo.slice`.
- Lower to other RMO ops first, then get legalized later in the same pass.

Legalization via the interface assumes all the output values are tensors, so
any operation like `rmo.mo.transfer` has to have a separate pattern. Device
handling is done via output annotation. After the pass, all ops are in `mo`
(TODO: verify).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/RMODialect/Transforms/LegalizeRMOOps.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/RMODialect/Transforms/RMOTransforms.td>

## MO passes

Pass definitions:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### AddFallbackShapeFunctions

#### Description

Materializes fallback shape functions for `mo` ops that still have unresolved
output parameters, ensuring every op has a shape function or an explicit
no-fallback declaration.

#### Dialects

Consumes and produces `mo` (GraphOp), using `kgen` shape-parameter attrs and
Mojo library metadata for `mo.custom`.

#### Additional notes

AddFallbackShapeFunctions is the safety net for MO shape inference: it ensures
that every MO op that still has unresolved output parameters gets a shape
function call inserted so shape parameters are materialized before lowering.
Its job is to make sure there are no remaining "unmaterialized" output param
decls.

For each `mo` op that:

- has the output param decls empty, meaning, params are already materialized -
  nothing to do.
- has unmaterialized dims and has a fallback rule - dispatch to a fallback
  builder to create shape computation and insert it before the op. The op is
  left intact and the implementation is handled by kernel selection as usual.
- has unmaterialized dims and is a custom op - query `MojoLibraryAnalysis` to
  find a registered shape function and build and invocation using those kernel
  params.
- needs no fallback (e.g., matmul, static broadcast) - nothing to do.

TODO: check phasing issue
<https://linear.app/modularml/issue/GEX-886/graphcompiler-make-symbolic-optimization-not-phase-issue-with-constant>.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/AddFallbackShapeFuncs.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### MergeDuplicateShapeMaterializations

#### Description

Deduplicates shape materialization ops so identical ops with the same operands
share a single defining instance and parameters.

#### Dialects

Consumes and produces `mo` (GraphOp), using `kgen` parameter metadata and some
`mogg` utilities.

#### Performance

Reduces redundant shape-parameter construction and improves CSE effectiveness
(semantically identical ops, but different parameters references).

#### Additional notes

Finds shape‑materialization ops that define all their parameters. If two such
ops have the same op name + same operands, it keeps one and deletes the rest
(remaps references using `ParameterCollector` analysis and building a def-use
graph).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/MergeDuplicateShapeMaterializations.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### InvokeShapePackingFunctions

#### Description

Evaluates packing-related `mo.invoke_shape_func` calls using a C++ shape
re-implementation and materializes concrete parameter declarations.

#### Dialects

Consumes and produces `mo` (GraphOp), with `kgen` parameter ops for emitted
declarations.

#### Performance

Precomputes packing shapes at compile time when possible, avoiding runtime
shape computation.

#### Additional notes

Pattern fusion may produce packing kernels (e.g., matmul B packing). The inputs
for kernels with packing are constructed via a sequence of shape functions +
`mo.layout_transform`. When the target is known, some of those shape functions
for packing can be resolved statically. The pass targets specific ones like
matmul/conv filter packing (matched by op name) and emits `kgen.param.declare`
for inferred dims. Optionally inserts sanity-check assertions.

When the target attribute is not present in the graph, the pass initializes
llvm native target and pull the information from current host.

Another instance of shape propagation (with CSE and canonicalization) is run
right after the pass, so that the newly introduced declarations are pushed
through the graph.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/InvokeShapePackingFunctions.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### ValidateDevices

#### Description

Validates that tensor dtypes in graph inputs/outputs are supported by the
target device architecture.

#### Dialects

Consumes and produces `mo` (GraphOp).

#### Additional notes

In preparation for MO to MOGG lowering, acts as correctness gate. Runs after
pattern fusion, so that all the potentially introduced tensors or device
annotation changes are settled.

The limitations are only put on CPU targets:

- no support for fp8.
- no support for fp16/bf16 on arm (for others, the types allowed for casting).
- no support for float types with size < 32bit on x86 and arch.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/ValidateDevicesPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### ConstantFold

#### Description

Identifies constant subgraphs and executes them at compile time using the
existing compilation pipeline, replacing them with `mo.constant` results.

#### Dialects

Runs on `mo` modules (`ModuleOp`) and drives MO to MGP lowering internally.

#### Performance

Reduces runtime work by precomputing constant regions; trades extra compile time
for smaller runtime graphs.

#### Additional notes

ConstantFold is graph‑level constant subgraph evaluation. It finds subgraphs
whose outputs are fully determined by constants, lowers and executes them, then
replaces their outputs with mo.constant values in the original graph.
The compilation consists of a nested MO->MGP pipeline with constant folding
disabled to avoid recursion and a MGP to MEF. All kernels are forced to CPU with
sync execution. The replacement converts the computed results back into
`mo.constant` (or constraints) and replaces the uses.

The pass is run each time when the pipeline introduces or exposes new constant
subgraphs.

Depends on `ParameterCollector` analysis.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/ConstantFold.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### HoistParameterExpressions

#### Description

Hoists complex parameter expressions out of tensor types into explicit
`kgen.param.declare` aliases and rewrites types to reference the new params.

#### Dialects

Consumes and produces `mo` (GraphOp) with `kgen` parameter attrs/ops and `mosh`
dependency.

#### Additional notes

Takes parameter expressions embedded inside tensor shapes (e.g.,
`KGEN::ParamOperatorAttr` like `mul(D0, D1)`) and hoists them into named
parameters. This is done to avoid repeated/hidden expressions inside types and
to make later passes simpler. Introduces `mo.rebind` on inputs/outputs to
preserve original types, creates one alias per unique param expression per
region, and folds consecutive rebinds.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/HoistParameterExpressionsPass.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### ElideRedundantMutableOps

#### Description

Eliminates redundant `mo.mutable.load` and `mo.mutable.store` ops when their
chains prove the operations are unnecessary.

#### Dialects

Consumes and produces `mo` (GraphOp), using chain semantics.

#### Performance

Removes unnecessary memory round-trips.

#### Additional notes

Elides a load when its input chain comes from a store to the same buffer and
the load is the sole chain user. Same for the stores.

The pass is only enabled together with `mogg` fusion (TODO: see notes).

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/ElideRedundantMutableOps.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### HoistConstantSubgraphsIntoParent

#### Description

Hoists callsite-invariant, constant-derived ops from subgraphs into their
parent graphs so they can be further hoisted to init or constant-folded.

#### Dialects

Runs on `mo` modules (`ModuleOp`) and uses `mo.call` call graphs.

#### Performance

Reduces repeated work inside subgraphs and unlocks EICM/constant folding of
invariant computations.

#### Additional notes

If a subgraph is called multiple times with the same SSA value for some
arguments, then any ops derived solely from those invariant args can be hoisted
to the callsite. The pass builds a call graph, identifies callsite-invariant
args and runs a dataflow analysis to find invariant-derived ops. The dataflow
analysis is a non-standard small implementation that depends on
`ParametricTopologyAnalysis` to reason about providers/users in the graph. In
operates on two-point lattice (const/non-const), and thus, is cheap and easy.

The optimization is performed when there are multiple callsitest. It ignores
cycles and treats "constant" as invariant across callsitets.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/HoistConstantSubgraphsIntoParent.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### InferLayouts

#### Description

Infers tensor layouts in `mo.graph` using layout constraints from ops like
`mo.conv`, `mo.conv_transpose`, and `mo.matmul`, then annotates tensor types
with the inferred layout attributes.

#### Dialects

Consumes and produces `mo` (GraphOp).

#### Performance

Enables layout-aware lowering and prevents later layout mismatches; can reject
unsupported conv ranks early.

#### Additional notes

The pass walks the mo.graph and adds hard layout constraints (seed constraints)
where the op semantics require a particular layout (e.g., `mo.conv` with a rank
of 4 in NHWC format). The second stage propagates those constraints through ops
that keep the input/output layout identical (the layout is pushed through users
and defining op). The inferred layouts are then assigned to tensors
(`TensorType`) - used for kernel selection and affects view fusion. `mo.graph`
is updated to account for the changed signatures when necessary. Emits errors
on conflicting constraints.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/InferLayouts.cpp>

### CanonicalizeChainSequences

#### Description

Optimizes linear (i.e. unique input/output) chains of side-effecting
`mo.mutable` ops by breaking chains that serialize unrelated buffers.
The pass rewrites sequences where independent buffer mutations are ordered by
the same chain, splitting the chain so later operations can start from an
earlier chain value when safe, while preserving ordering for dependent ops.

Implementation lives in `CanonicalizeBufferOpSequences` rewrite pattern and the
canonicalization patterns for `ChainCreateOp` (which is currently just
flattening).

#### Dialects

Consumes and produces `mo` (GraphOp) with chain semantics (`mo.chain.create`,
`mo.mutable.*` ops implementing `MOMutableOpInterface`).

#### Performance

Reduces unnecessary serialization (e.g., splitting unnecessary intermediate
chains) of side-effecting ops on different buffers, which can improve
parallelism and scheduling freedom.

#### Additional notes

Chains are introduced at the IR construction time (e.g., by the Graph API).
Implemented as a greedy rewrite over `GraphOp` with an increased iteration cap
for long chains; may skip ops that have multiple in/out chains. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/CanonicalizeChainSequences.cpp>

### PropagateShapes

#### Description

Propagates and simplifies shape-parameter logic through the graph. It evaluates
shape computations in the parameter domain, materializes shape functions from
parameter values, and simplifies parameter bindings via substitution and
constraint solving to rebind parameter attributes/types as it walks the graph.
It also normalizes cross-graph call operands to match callee shape symbols and
replaces `mo.shape_of` with constant shape values when possible.

At a high-level these are the steps taken:

1) Topologically sort ops with SSA + parameter dependencies.
2) Worklist walk: materialize shape functions, staticize data-dependent
   (requiring reading runtime values from input tensors) shape inputs, and
   fold parameter expressions/ranges.
3) Rebind/replace parameterized types and attrs, update graph signature and
   output bindings.
4) DCE unused `kgen.param.declare` via use-def graph.
5) Normalize `mo.shape_of` and cross-graph `mo.call` operands.

#### Dialects

Consumes and produces `mo` (GraphOp) and `kgen` parameter IR, with shape support
from `mosh` and `mo.shape` ops.

#### Performance

Reduces shape-related SSA and parameter dependencies, enabling more CSE and
canonicalization, and improving compile-time simplifications of shape logic.

#### Additional notes

Used in the shape inference pipeline and in the pattern-fusion path after
`InvokeShapePackingFunctions`. The pass relies on `ParameterCollector` analysis.
It performs a parametric topological sort (via `ParametricTopologyAnalysis`) so
the worklist visits ops only after their SSA and parameter dependencies are
available, which is required for consistent rebinding and folding. It also
builds a `KGEN::ParameterUseDefGraph` to drive dead `kgen.param.declare`
elimination, and uses `KGEN::ParameterEvaluator` for substitution/constraint
solving. Shape materialization depends on the `ShapeMaterialization` (lets ops
generate the IR needed to compute their output shape parameters) and
`Staticization` (convert tensors into symbolic parameter expressions) interfaces
to evaluate shape logic. Dynamic ranks are not supported. Staticizing only works
for tensors of rank less than 2. CSE for parameter expressions doesn't cross
region boundaries. The pass is the basis for shape inferencing. The rest is
canonicalization and CSE, plus `KGEN::VerifyParameters` (TODO). Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/PropagateShapesPass.cpp>

### SymbolicOptimization

#### Description

Applies domain-specific optimizations using symbolic information in the graph,
typically produced by shape inference. The pass rewrites and canonicalizes
operation sequences to simpler or more efficient forms, and can perform
fast-math style rewrites under configurable assumptions.

#### Dialects

Consumes and produces `mo` (GraphOp) with symbolic shape data from `mosh` and
parameter expressions from `kgen`.

#### Performance

Reduces redundant computation and enables later fusion by simplifying symbolic
subgraphs and normalizing shapes/reshapes/transposes. Also converts operations
to forms that have better kernel implementations.

#### Additional notes

Runs twice in the pipeline:

1. After initial shape inference and constant folding, before adding fallback
   shape functions.
2. After adding fallback shape functions, to optimize newly materialized
   operations.

The pass is organized in stages that allow for shape-driven algebraic
simplification. There are "core" patterns that roughly include the following
groups of transformations:

- data-layout "swimming" (moving shape-changing ops through other ops to either
  facilitate reuse or enable fusion/canonicalization; e.g., sink transposes
  through unary/binary ops)
- Contraction ops normalization and simplification
- split/concat canonicalization, including gather/squeeze patterns and similar
- Algebraic simplification with simple expressions folding (e.g., x+0 -> x) and
  optional arith-unstable patterns under configuration flags
  (`symbolic-ffast-math` and `symbolic-no-undefined-math`).

The pass applies these core patterns multiple times intertwining with forward
and backward mathematical canonicalizations, together with splice <-> split
conversions. Those decompose and compose certain operations/sequences (e.g.,
rsqrt <-> 1 / sqrt) to try and give more context for canonicalization patterns
and core patterns to apply.

Some limitations/requirements:

- Requires symbolic shape information (known shapes)
- Some patterns are conservative (require single-use constraints)
- Fast-math optimizations may change floating-point semantics
- Top-down traversal may miss some optimization opportunities
- Reliance on the sequence of patterns application

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/SymbolicOptimization.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/Transforms/MOTransforms.td>

### PatternFusion

#### Description

Fuses specific operator patterns into hand-fused kernels or more specialized
ops. It applies a set of pattern rewrites (e.g., matmul/conv const folding,
layer norm and RMS norm fusions, concat/split axis-0 rewrites, and multi-head
attention fusion) to replace multi-op subgraphs with optimized implementations
or canonical forms.

#### Dialects

Consumes and produces `mo` (GraphOp) and some `mogg` ops, and depends on `kgen`,
`mosh`, and `mogg`. Technically, the pass lives in the `mogg` transforms,
however it runs on `mo` ops in the mo-pipeline, before the advanced-fusion
lowering (`MOToMAP`).

#### Performance

Enables higher-level kernel fusion and reduces dispatch overhead by replacing
common subgraphs with hand-fused kernels or more efficient ops.

#### Additional notes

Implemented as a greedy rewrite over `GraphOp`, using `PatternFusionOptions` to
toggle MHA matching and device targeting. After rewrites, the pass calls
`MO::synchronizeMetadataUpdates` to keep metadata consistent with the
transformed graph.

PatternFusion patterns use the device-aware rewrite wrappers
(`MO::DeviceAwareRewritePattern` + `MO::DeviceAwareRewriter`) so each pattern
collects device annotations from all matched ops, rejects conflicting device
sets, and stamps the chosen device onto newly created fused ops. This lets the
pattern implementations perform device-specific checks and still emit
device-consistent replacements. There can (and are) copies of very similar
passes with device specialization. Device-aware rewriter definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MODialect/MODevices.h#L98>.

Some patterns emit `mo.custom` calls that route into the MOGG Kernel API layer
(TODO: check the lowering paths and how they affect the fusion); other patterns
rewrite to `mo` ops or `mogg` ops like `mogg.concat` and `mogg.split`
(interestingly, there are kernels for both in the kernel API, but those are
registered for `mo` and are unused in case pattern fusion is on). The MOGG
Kernel API layer is a registry of hand-fused kernels exposed via
`@extensibility.register(...)` symbols, which `mo.custom` uses as the dispatch
key at lowering time. MOGG Kernel API definitions (docs-by-source):
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/max/kernels/src/graph_compiler/builtin_kernels/attention.mojo#L231>

Most complexity is in the matching of complex patterns such as MHA. The
matching has restrictions, e.g., single-use for intermediate values.

The pass is gated by device/layout inference & validation.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MODialect/Transforms/PatternFusion.cpp>

## MOGG passes

### JitCompileKernels

#### Description

Takes Mojo source text emitted by the compiler (stored as a module resource),
parses it, runs the KGEN pipelines (LIT -> KGEN -> elaborated KGEN -> LLVM), and
replaces the MOGG::CompiledLoadOp resource with the compiled binary.

#### Dialects

Uses `lit` to parse Mojo sources, and depends on `m`, `mogg`, `kgen`, `pop`,
`hlcf`, debug info, as well as target dialects (`llvm`, `nvvm`, `rocdl`,
`index`).

#### Performance

Generates compiled kernel blobs and enables later lowering stages to reference
compiled code instead of high-level kernel bodies.

#### Additional notes

JitCompileKernels is the backbone of the JIT compilation pipeline (last stage
of gc-pipeline) that converts `mogg` kernel regions into executable code. The
rest of the pipeline runs on the surrounding IR that has references to the
compiled blobs. The pass looks up the source code in the Mojo resource blob,
parses Mojo source into MLIR module with LIT parser, runs KGEN pipelines
(generate library and the elaboration pipelines), lowers to llvm, optimizes
(-O3), and runs codegen (via ObjectCompiler). The binaries are then embedded
into the MEF as resources attached to the module load path.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/JitCompileKernels.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### HashModuleContents

#### Description

Computes a semantic hash of module contents and attaches it as an attribute on
KGEN generators, enabling cache keys and change tracking across kernel modules.

#### Dialects

Operates on `mogg`/`kgen` modules (`mlir::ModuleOp`) with symbol references.

#### Performance

Adds hashing overhead during compilation but enables downstream cache reuse and
deduplication based on semantic equivalence.

#### Additional notes

Not a part of the default pipeline. No usage currently. Builds a callgraph that
tracks symbol references in attributes/types, hashes generator bytecode with
normalized names/locations, and writes `semantic_hash` attributes. In that sense
it is "semantic", i.e., insensitive to non-semantic differences like
names/location. Uses `SymbolTableAnalysis` and `ModuleHasher`. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/HashModuleContents.cpp>
Implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Support/ModuleHasher.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

### NormalizeAllNames

#### Description

Normalizes symbol and parameter names in modules to deterministic, synthetic
names to improve deduplication and stability across compiles.

#### Performance

Adds light renaming overhead but increases cache hits and reduces spurious
differences in generated kernels.

#### Additional notes

The implementation is used in the framework frontend as a cache-key
normalization step (not the pass itself). Renames symbols to `__S<index>`, then
walks operations in parallel to rename parameters to `__P<index>` and update all
symbol/parameter references. Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/lib/GraphCompiler/MOGGDialect/Transforms/NormalizeAllNames.cpp>
Implementation:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/Mojo/lib/KGENDialect/KGENUtils.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/GraphCompiler/include/GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.td>

## KGEN passes

### VerifyParameters

#### Description

Validates KGEN parameter declarations and uses, detecting mismatches,
conflicting declarations, and parameter dependency cycles. Optionally performs
trivial parameter propagation and rebinding after verification.

#### Dialects

Runs on `kgen` modules and depends on `kgen` parameter IR and the LIT symbol
table evaluation context for parameter resolution.

#### Additional notes

Builds a `ParameterUseDefGraph` per declaration region and verifies it against a
`LIT::LITSymTabEvaluationContext`, using a shared `ParameterCollector` cache and
`SymbolTableAnalysis`. If `simplify` is enabled, it propagates trivial parameter
definitions, rebinds attributes/types/locations, removes `kgen.param.declare`
ops, and peels redundant `kgen.rebind` ops.

Source:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/Mojo/lib/Transforms/VerifyParameters.cpp>
Pass definition:
<https://github.com/modularml/modular/blob/b71cbf8b216acc98960181ebedcd1a1a627b2288/Mojo/include/Mojo/KGENPasses.td>
