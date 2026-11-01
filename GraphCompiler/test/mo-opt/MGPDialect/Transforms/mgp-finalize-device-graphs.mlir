// RUN: mo-opt %s --mgp-finalize-device-graphs --allow-unregistered-dialect --split-input-file | FileCheck %s

// The `mgp.device_graph.collective` op of an `isDeviceGraph` model is lifted
// into a `mgp.device_graph.collective.create` wrapped in
// a `mogg.routine.async` (so it compiles via the kernel path), plus
// a bare `mgp.device_graph.collective.execute` that replays the graph bundle;
// `mef.output` is rewired to the execute op's outputs.

// CHECK-LABEL: mgp.model @dg
// CHECK: execute (%[[CH0:[a-zA-Z0-9_]+]]: !mef.chain, %[[CTX:[a-zA-Z0-9_]+]]: !mgp.context, %{{.+}}: !mgp.opaque<"weights_registry">, %[[GPU:[a-zA-Z0-9_]+]]: !mgp.device_context<<"gpu", 0>>, %[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>)

// CHECK: %[[GRAPH:[a-zA-Z0-9_]+]] = mogg.routine.async (%[[CTX]], %[[GPU]], %[[IN]]) : (%{{.+}}: !mgp.context, %{{.+}}: !mgp.device_context<<"gpu", 0>>, %{{.+}}: !mgp.tensor<2x3xf32, <"gpu", 0>>)  -> !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>> {
// CHECK: mogg.async.unpack

// CHECK: %[[CACHE:[a-zA-Z0-9_]+]] = mgp.device_graph.cache %{{.+}} : !mgp.device_graph.cache
// CHECK: %[[CGRAPH:[a-zA-Z0-9_]+]] = mgp.device_graph.collective.create %{{.+}} cache(%[[CACHE]] : !mgp.device_graph.cache) graph(%{{.+}} : !mgp.device_context<<"gpu", 0>>) builder(%b0) inputs(%{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK: mgp.device_graph.yield %{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mogg.async.pack
// CHECK: mogg.output

// CHECK: %{{[a-zA-Z0-9_]+}}, %[[RES:[a-zA-Z0-9_]+]] = mgp.device_graph.collective.execute %[[GRAPH]]
// CHECK: mef.output %[[CH0]], %[[RES]] : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NOT: mgp.device_graph.collective builder

mgp.model @dg device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A model WITHOUT `isDeviceGraph` is left untouched: no create/execute is
// introduced and the staging op stays.

// CHECK-LABEL: mgp.model @not_dg
// CHECK: mgp.device_graph.staging
// CHECK-NOT: mgp.device_graph.collective.create

mgp.model @not_dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    mgp.device_graph.staging.yield %in_t : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// Device graphs have no chain concept. When the collective body yields a chain
// (host-side sequencing) in addition to its value output, finalize must
// (a) keep `!mef.chain` out of the `mgp.device_graph` signature entirely and
// (b) give `mgp.device_graph.collective.execute` its own fresh completion chain
// result, then rewire the collective op's chain result to it.

// CHECK-LABEL: mgp.model @dg_chain
// The graph signature is chain-free on BOTH sides (inputs and results).
// CHECK: mgp.device_graph.collective.create %{{.+}} cache(%{{.+}} : !mgp.device_graph.cache) graph(%{{.+}} : !mgp.device_context<<"gpu", 0>>) builder(%b0) inputs(
// CHECK: } : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>
// The execute op produces a completion chain (result #0) + the value output; the
// collective op's chain result is rewired to that chain (fed to `mef.output`).
// CHECK: %[[CHAIN:[a-zA-Z0-9_]+]], %[[RES:[a-zA-Z0-9_]+]] = mgp.device_graph.collective.execute
// CHECK: mef.output %[[CHAIN]], %[[RES]] : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NOT: mgp.device_graph.collective builder
mgp.model @dg_chain device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %c0:2 = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %r = "test.identity"(%in) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    // Host-side "done" chain, yielded as a graph output today; must be dropped.
    %done = "test.make_chain"() : () -> !mef.chain
    mgp.device_graph.collective.yield %done, %r : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %c0#0, %c0#1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// `mgp-compute-device-graph-dependencies` re-sources an in-body device context
// from the recording builder through an unrealized cast, expecting the consumer
// to be rewritten to take the builder (as `mgp.buffer.alloc` is). A consumer
// that genuinely needs a device context -- `mgp.buffer.bulk_slice`, which only
// does host-side sub-buffer math -- leaves the cast live, and an unrealized cast
// cannot be emitted to Mojo. Finalize replaces it with the real accessor op.

// CHECK-LABEL: mgp.model @dg_builder_context
// CHECK: mgp.device_graph.collective.create
// The cast lives in a `dependencies` body, so it is that body's builder arg --
// not the create body's -- that the accessor op reads.
// CHECK: mgp.device_graph.dependencies builder(%{{.*}})
// CHECK-NEXT: ^bb0(%[[DB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK: %[[CTX:[a-zA-Z0-9_]+]] = mgp.device_graph.device_context %[[DB]] : !mgp.device_context<<"gpu", 0>>
// CHECK: mgp.buffer.bulk_slice %{{.+}}, %[[CTX]] offsets
// CHECK-NOT: unrealized_conversion_cast
mgp.model @dg_builder_context device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.buffer<<"gpu", 0>>) -> (!mef.chain, !mgp.buffer<<"gpu", 0>>) {
  %out = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node, %r = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %dev = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %off = "test.offset"() : () -> index
      %size = "test.size"() : () -> index
      %slice = mgp.buffer.bulk_slice %in, %dev offsets(%off) sizes(%size) : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
      mgp.device_graph.dependencies.yield %slice : !mgp.buffer<<"gpu", 0>>
    } : !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.collective.yield %r : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.buffer<<"gpu", 0>>
}

// -----

// A `builder -> device context -> builder` round trip (what
// `mgp-generate-build-graph-stubs` leaves behind when it rewrites
// `mgp.buffer.alloc` into the builder-native `mgp.device_graph.alloc`) must fold
// away entirely. Converting the first cast to `mgp.device_graph.device_context`
// before folding would strand the inverse cast, so no accessor op may appear.

// CHECK-LABEL: mgp.model @dg_builder_round_trip
// CHECK: mgp.device_graph.collective.create
// CHECK: mgp.device_graph.dependencies
// CHECK-NEXT: ^bb0(%[[DB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK: mgp.device_graph.alloc %{{.+}}, %{{.+}}, %[[DB]], align 256
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: mgp.device_graph.device_context
mgp.model @dg_builder_round_trip device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.buffer<<"gpu", 0>>) -> (!mef.chain, !mgp.buffer<<"gpu", 0>>) {
  %out = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node, %r = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %dev = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %back = builtin.unrealized_conversion_cast %dev : !mgp.device_context<<"gpu", 0>> to !mgp.device_graph.builder
      %bch = "test.make_chain"() : () -> !mef.chain
      %size = "test.size"() : () -> index
      %buf = mgp.device_graph.alloc %bch, %size, %back, align 256 : !mgp.buffer<<"gpu", 0>>
      mgp.device_graph.dependencies.yield %buf : !mgp.buffer<<"gpu", 0>>
    } : !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.collective.yield %r : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.buffer<<"gpu", 0>>
}

// -----

// A collective body may yield one of the values it captures from outside (a
// `mgp.device_graph.dependencies` body result that is itself an outer input).
// The finalize pass remaps that capture to the create op's per-region input
// arg when cloning the body, so the `mgp.device_graph.yield` and any other uses
// end up referencing the right value.

// CHECK-LABEL: mgp.model @dg_passthrough
// CHECK: mgp.device_graph.collective.create %{{.+}} cache(%{{.+}} : !mgp.device_graph.cache) graph(%{{.+}} : !mgp.device_context<<"gpu", 0>>) builder(%b0) inputs(
// CHECK: mgp.device_graph.yield %{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mgp.device_graph.collective.execute
// CHECK-NOT: mgp.device_graph.collective builder
mgp.model @dg_passthrough device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A mut-marked input (the lowered form of !mo.buffer) flows through finalization
// with its type spelled verbatim at every boundary: the compiled region's
// signature, the create op's inputs and block args, the graph signature, and the
// execute op's inputs.

// CHECK-LABEL: mgp.model @dg_mut_input
// CHECK: execute (%[[CH0:[a-zA-Z0-9_]+]]: !mef.chain, %[[CTX:[a-zA-Z0-9_]+]]: !mgp.context, %{{.+}}: !mgp.opaque<"weights_registry">, %[[GPU:[a-zA-Z0-9_]+]]: !mgp.device_context<<"gpu", 0>>, %[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>, mut>)
// CHECK: mogg.routine.async (%[[CTX]], %[[GPU]], %[[IN]]) : (%{{.+}}: !mgp.context, %{{.+}}: !mgp.device_context<<"gpu", 0>>, %{{.+}}: !mgp.tensor<2x3xf32, <"gpu", 0>, mut>)  -> !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>, mut>) -> !mgp.tensor<2x3xf32, <"gpu", 0>, mut>>
// CHECK: mgp.device_graph.collective.create %{{.+}} cache(%{{.+}} : !mgp.device_graph.cache) graph(%{{.+}} : !mgp.device_context<<"gpu", 0>>) builder(%b0) inputs(%{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>, mut>) {
// CHECK: mgp.device_graph.yield %{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
// CHECK: } : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>, mut>) -> !mgp.tensor<2x3xf32, <"gpu", 0>, mut>>
// CHECK: mgp.device_graph.collective.execute
// CHECK-NOT: mgp.device_graph.collective builder

mgp.model @dg_mut_input device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>, mut>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, mut>) {
  %out = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield %in : !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
}

// -----

// A multi-device `mgp.device_graph.collective` with no stable-address slots
// lowers to a single `mogg.routine.async` that builds all graphs
// via `mgp.device_graph.collective.create`, followed by a host-side
// `mgp.device_graph.collective.execute`.

// CHECK-LABEL: mgp.model @dg_collective
// CHECK: execute (%[[CH:[a-zA-Z0-9_]+]]: !mef.chain, %[[CTX:[a-zA-Z0-9_]+]]: !mgp.context, %{{.+}}: !mgp.opaque<"weights_registry">, %[[GPU0:[a-zA-Z0-9_]+]]: !mgp.device_context<<"gpu", 0>>, %[[GPU1:[a-zA-Z0-9_]+]]: !mgp.device_context<<"gpu", 1>>, %[[IN0:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>, %[[IN1:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 1>>)
// CHECK: %[[GRAPHS:[a-zA-Z0-9_]+]]:2 = mogg.routine.async (%[[CTX]], %[[GPU0]], %[[GPU1]], %[[IN0]], %[[IN1]]) : (%{{.+}}: !mgp.context, %{{.+}}: !mgp.device_context<<"gpu", 0>>, %{{.+}}: !mgp.device_context<<"gpu", 1>>, %{{.+}}: !mgp.tensor<2x3xf32, <"gpu", 0>>, %{{.+}}: !mgp.tensor<2x3xf32, <"gpu", 1>>)  -> (!mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>, !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 1>>) -> !mgp.tensor<2x3xf32, <"gpu", 1>>>)
// CHECK: %[[CACHE:[a-zA-Z0-9_]+]] = mgp.device_graph.cache %{{.+}} : !mgp.device_graph.cache
// CHECK: %[[CGRAPHS:[a-zA-Z0-9_]+]]:2 = mgp.device_graph.collective.create %{{.+}} cache(%[[CACHE]] : !mgp.device_graph.cache) graph(%{{.+}} : !mgp.device_context<<"gpu", 0>>) builder(%b0) inputs(%[[RI0:[a-zA-Z0-9_]+]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK:   mgp.device_graph.yield %[[RI0]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: }, graph(%{{.+}} : !mgp.device_context<<"gpu", 1>>) builder(%b1) inputs(%[[RI1:[a-zA-Z0-9_]+]] : !mgp.tensor<2x3xf32, <"gpu", 1>>) {
// CHECK:   mgp.device_graph.yield %[[RI1]] : !mgp.tensor<2x3xf32, <"gpu", 1>>
// CHECK: } : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>, !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 1>>) -> !mgp.tensor<2x3xf32, <"gpu", 1>>>
// CHECK: %{{[a-zA-Z0-9_]+}}, %[[OUTS:[a-zA-Z0-9_]+]]:2 = mgp.device_graph.collective.execute %[[GRAPHS]]#0, %[[GRAPHS]]#1 inputs(%[[IN0]], %[[IN1]] : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>, !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 1>>) -> !mgp.tensor<2x3xf32, <"gpu", 1>>> -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
// CHECK: mef.output %[[CH]], %[[OUTS]]#0, %[[OUTS]]#1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
// CHECK-NOT: mgp.device_graph.collective builder

mgp.model @dg_collective device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) {
  %c0, %c1 = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield %in0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
  }, <"gpu", 1> {
    mgp.device_graph.collective.yield %in1 : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0, %c1
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
}

// -----

// A host-placed tensor input (dispatch metadata) flows through finalization
// with its cpu placement spelled verbatim at every boundary; the graph device
// still resolves from the gpu output, not the host input.

// CHECK-LABEL: mgp.model @dg_host_input
// The host input is captured first (it is used textually before the gpu input
// in the body), so it lands before the gpu input in the create/execute input
// lists and the graph signature.
// CHECK: mogg.routine.async
// CHECK: mgp.device_graph.collective.create %{{.+}} cache(%{{.+}} : !mgp.device_graph.cache) graph(%{{.+}} : !mgp.device_context<<"gpu", 0>>) builder(%b0) inputs(%{{.+}} : !mgp.tensor<4xsi64>, %{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK: } : !mgp.device_graph<(!mgp.tensor<4xsi64>, !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>
// CHECK: mgp.device_graph.collective.execute
// CHECK-NOT: mgp.device_graph.collective builder

mgp.model @dg_host_input device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>, %h: !mgp.tensor<4xsi64, <"cpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    // The host input is read inside the body (would key the cache); keep it used.
    "test.use_host"(%h) : (!mgp.tensor<4xsi64, <"cpu", 0>>) -> ()
    mgp.device_graph.collective.yield %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// Stable address slots are declared on the collective rather than passed as
// values, so finalization needs no capture plumbing for them: the regions
// reach slots by index through the builder, the one block argument
// finalization already remaps. The declaration is copied onto the create op
// so emission can allocate the slots.

// CHECK-LABEL: mgp.model @dg_stable_slots
// CHECK: mgp.device_graph.collective.create
// CHECK-SAME: builder(%[[B0:[a-zA-Z0-9_]+]])
// CHECK: %[[P0:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.place %[[B0]][0], %[[IN0:[a-zA-Z0-9_]+]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: %[[S0:[a-zA-Z0-9_]+]] = mgp.tensor.extract.tensor_spec %[[IN0]]
// CHECK: mgp.device_graph.stable.tensor %[[P0]], %[[S0]] : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
// CHECK: builder(%[[B1:[a-zA-Z0-9_]+]])
// CHECK: %[[A0:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.get %[[B1]][0] : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
// CHECK: %[[S1:[a-zA-Z0-9_]+]] = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
// CHECK: mgp.device_graph.stable.tensor %[[A0]], %[[S1]] : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
// CHECK: {stableTypes = #mgp<stable_addr_array[<<2x3xf32, <"gpu", 0>>>]>}
// CHECK: mgp.device_graph.collective.execute
// CHECK-NOT: mgp.device_graph.collective builder

mgp.model @dg_stable_slots device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) {
  %c0, %c1 = mgp.device_graph.collective builder(%b) stable(
      !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    %p0 = mgp.device_graph.stable.place %b[0], %in0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %s0 = mgp.tensor.extract.tensor_spec %in0 : <2x3xf32, <"gpu", 0>>
    %t0 = mgp.device_graph.stable.tensor %p0, %s0 : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
    mgp.device_graph.collective.yield %in0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
  }, <"gpu", 1> {
    %a0 = mgp.device_graph.stable.get %b[0] : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
    %s1 = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %t1 = mgp.device_graph.stable.tensor %a0, %s1 : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
    mgp.device_graph.collective.yield %in1 : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0, %c1
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
}

// -----

// A tensor a region reads only for its shape is recorded on the create op by
// its flat input position. gpu:1's region takes gpu:0's tensor, but every use
// is a spec extraction, so that input (region 0 contributes one input before
// it) is shape-only; gpu:1's own tensor, whose buffer is read, is not.
//
// CHECK-LABEL: mgp.model @dg_shape_only_input
// CHECK: mgp.device_graph.collective.create
// CHECK: shapeOnlyInputs = array<i64: 1>

mgp.model @dg_shape_only_input device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) {
  %c0, %c1 = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield %in0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
  }, <"gpu", 1> {
    // By finalization the dependency pass has inlined the host bookkeeping:
    // gpu:0's tensor is read only for its shape, gpu:1's for its buffer.
    %spec = mgp.tensor.extract.tensor_spec %in0 : <2x3xf32, <"gpu", 0>>
    %ch = mogg.chain.dummy()
    %buf = mgp.tensor.extract.buffer %in1 : <2x3xf32, <"gpu", 1>>
    %r = mgp.tensor.create %ch, %buf, %spec : !mgp.tensor<2x3xf32, <"gpu", 1>>
    mgp.device_graph.collective.yield %r : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
  mef.output %ch0, %c0, %c1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
}
