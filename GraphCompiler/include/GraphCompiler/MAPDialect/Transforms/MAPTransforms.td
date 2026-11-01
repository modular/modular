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

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def FuseKernels : Pass<"map-fuse-kernels", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels by inlining producers into consumers.";
  let description = [{
    Runs the advanced fusion patterns over the MAP dialect, reducing the number
    of `map` kernels by inlining producers into consumers.

    Anchored on `mo.graph` (which is `IsolatedFromAbove` and always a direct
    child of the module), so the pass never walks the imported kernel decls or
    the insides of kernels it already visited, and independent graphs run in
    parallel.

    Repeats epilogue, elementwise, view, then prologue fusion until a round
    fuses nothing, since folding one link of a chain can expose the next.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];

  let statistics = [
    Statistic<"numKernelsFused", "num-kernels-fused", "Number of kernels fused">
  ];
}

def FuseEpilogue : Pass<"map-fuse-epilogue", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels via EpilogueFuser only. For testing only.";
  let description = [{
    Runs `EpilogueFuser` alone so a lit test can exercise it in isolation.
    Not part of `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseHorizontal : Pass<"map-fuse-horizontal", "MO::GraphOp"> {
  let summary = "Fuse sibling MAP kernels that share an input.";
  let description = [{
    Merges two `map.iter.elementwise` ops over the same index space that share an
    input into one multi-output kernel, so the shared operand is loaded once
    rather than by each kernel. Pairs siblings, where the other fusers pair a
    producer with its consumer. Also moves a consumer into its producer's
    epilogue when the producer's result has other readers, which output
    fusion leaves alone; a consumer that reads a strided window of the result
    runs there under a `map.index.slice.select`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseReshapeEpilogue : Pass<"map-fuse-reshape-epilogue", "MO::GraphOp"> {
  let summary =
      "Fuse MAP kernels via ReshapeEpilogueFuser only. For testing only.";
  let description = [{
    Runs `ReshapeEpilogueFuser` alone so a lit test can exercise it in
    isolation. Not part of `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseElementwise : Pass<"map-fuse-elementwise", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels via ElementwiseFuser only. For testing only.";
  let description = [{
    Runs `ElementwiseFuser` alone so a lit test can exercise it in isolation.
    Not part of `buildMapPipeline`.
  }];

  let options = [
    Option<"largeTensorElementThreshold", "large-tensor-element-threshold",
           "int64_t", /*default=*/"4096",
           "Tensor operands with more static elements than this (or with an "
           "unknown element count) are considered large; a kernel with more "
           "than one large tensor operand is not duplicated into multiple "
           "consumer subgraphs.">,
    Option<"maxFusionSetSize", "max-fusion-set-size", "uint64_t",
           /*default=*/"64",
           "Cap on the number of kernels fused into one consumer, bounding the "
           "duplication of a cheap multi-use producer across an elementwise "
           "chain with fan-out.">,
  ];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseViews : Pass<"map-fuse-views", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels via ViewFuser only. For testing only.";
  let description = [{
    Runs `ViewFuser` alone so a lit test can exercise it in isolation. Not
    part of `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];

  let statistics = [
    Statistic<"numKernelsFused", "num-kernels-fused", "Number of kernels fused">
  ];
}

def FusePrologue : Pass<"map-fuse-prologue", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels via PrologueFuser only. For testing only.";
  let description = [{
    Runs `PrologueFuser` alone so a lit test can exercise it in isolation.
    Not part of `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseReductions : Pass<"map-fuse-reductions", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels into rowwise reductions. For testing only.";
  let description = [{
    Runs the reduction fuser alone. A producer only a `map.iter.rowwise`
    reads is cloned at each load of its result. A consumer over the per-row
    shape that reads a per-row result at its own index becomes row-level
    ops, and its result a rowwise output.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];

  let statistics = [
    Statistic<"numKernelsFused", "num-kernels-fused", "Number of kernels fused">
  ];
}

def FuseInput : Pass<"map-fuse-input", "MO::GraphOp"> {
  let summary = "Unconstrained generic input fusion. For testing only.";
  let description = [{
    Runs generic input fusion with no wrapper constraint: every eligible
    producer/consumer edge fuses, subject only to the shared correctness
    constraints. Not part of `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseOutput : Pass<"map-fuse-output", "MO::GraphOp"> {
  let summary = "Unconstrained generic output fusion. For testing only.";
  let description = [{
    Runs generic output fusion with no wrapper constraint: every eligible
    epilogue edge fuses, subject only to the shared correctness constraints.
    Not part of `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def OutlineStructuredKernelBodies
    : Pass<"map-outline-structured-kernel-bodies", "mlir::ModuleOp"> {
  let summary =
      "Outline structured MAP kernel bodies into `map.iter.opaque` regions.";
  let description = [{
    Wraps each structured MAP loop (`map.iter.elementwise`, `map.iter.rowwise`) in
    a `map.iter.opaque` region so sequential setup (e.g. strided-view lifting)
    can live outside the structured loop body. The loop's own shape and body
    move into a nested `map.iter.elementwise.body` or `map.iter.rowwise.body`,
    which reaches the wrapper's inputs/outputs through its captures.
    After outlining, nothing inspects the inside of the `map.iter.opaque` again.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
  ];
}

def LoopInvariantViewMotion : Pass<"map-livm", "MO::GraphOp"> {
  let summary =
      "Hoist chains of index-transform ops out of a structured loop body.";
  let description = [{
    Runs after `OutlineStructuredKernelBodies`, over every `map.iter.opaque`
    in the graph, including ones nested in control-flow regions. It anchors on
    `mo.graph` because a pass adaptor does not descend into an `mo.while`,
    `mo.if` or `mo.parallel` body. The rewrite is mandatory: an unhoisted
    index-transform op has no `LowerableToMOGGInterface` and strands
    `map-to-mogg`.

    For each `map.load` in a `map.iter.elementwise.body`, walks back from its index
    through any chain of `MAP::IndexTransformOpInterface` ops (today:
    `map.index.slice`) to the loop's own index, hoists each hop into a "stride"
    op (today: `map.index.slice.stride`) in the enclosing `map.iter.opaque`, so
    it runs once per kernel invocation rather than once per index, and rewrites
    the load to index the hoisted tensor with the untransformed index. Loads
    that hoist an SSA-identical chain share one hoisted op.

    Invariant: only index-transform ops that feed a store index survive.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
  ];
}

def RaiseBackContiguousViews
    : Pass<"map-raise-back-contiguous-views", "MO::GraphOp"> {
  let summary = "Raise unfused slice/reshape view kernels back to MO/MOGG ops.";
  let description = [{
    Runs after `FuseKernels`, before `OutlineStructuredKernelBodies`. Raises
    unfused `map.iter.elementwise` view kernels back into ops `MOGGToMGP` lowers
    without a copy: a contiguous slice becomes `mogg.slice`
    (`mgp.tensor.slice`), and a reshape becomes `mo.static.reshape`. Like
    `mogg-fuse-views`, only slice and reshape, not broadcast/transpose.
  }];

  let dependentDialects = [
    "M::MO::MODialect",
    "M::MOGG::MOGGDialect",
  ];

  let statistics = [
    Statistic<"numSlicesRaised", "num-slices-raised",
              "Number of contiguous slice kernels raised back">,
    Statistic<"numReshapesRaised", "num-reshapes-raised",
              "Number of reshape kernels raised back">
  ];
}

def FuseMutableLoads : Pass<"map-fuse-mutable-loads", "MO::GraphOp"> {
  let summary = "'Fuse' mo.mutable.load ops with map.iter.elementwise ops.";
  let description = [{
    Will perform 'fusion' of a mo.mutable.load op with a map.iter.elementwise op.
    This amounts to:
      1. Replacing the value semantic tensor created by the mutable.load op
         with the buffer semantic tensor
      2. Making the input chain of the load op an input of the
         map.iter.elementwise op
      3. Adding the output chain of the mutable.load op to the output of the
         map.iter.elementwise op
  }];
  let dependentDialects = ["M::MAP::MAPDialect"];
}

def AllocateAccumulatingBuffers
    : Pass<"map-allocate-accumulating-buffers", "MO::GraphOp"> {
  let summary = "Give every fused-output epilogue a real store pointer.";
  let description = [{
    Runs after `FuseKernels`. Sets each `map.iter.epilogue`'s
    `$intermediateTensor` so `MAPToMOGG` always has a real base pointer:

    - Normally the pointer source is the tensor the epilogue stores into.
    - For a mixed-precision accumulating kernel (`containsMatmulOrConvOrConcat`
      whose epilogue changes dtype), the kernel must accumulate in the pre-cast
      dtype before the lambda casts and stores. A fresh accumulation buffer of
      that dtype is allocated as an extra kernel output and used as the pointer
      source; the store target stays the real (post-cast) output.
  }];
  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MO::MODialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
