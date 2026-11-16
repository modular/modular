// RUN: mo-opt %s --mgp-outline-device-graph-dependencies \
// RUN:     --mogg-merge-identical-funcs --allow-unregistered-dialect \
// RUN:     --split-input-file | FileCheck %s

// `mgp-outline-device-graph-dependencies` moves the recording body of each
// `mgp.device_graph.dependencies` op with no body results into a module-level
// `mogg.func` and replaces the body with a single `mogg.call`. The body
// references enclosing SSA directly (the op is not `IsolatedFromAbove`), so
// what it captures becomes a trailing stub argument forwarded at the call
// site. The stub is a normal function, so identical recordings merge below.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

// A callee for the recording body, i.e. the stub `OutlineRoutines` outlined
// the unit's kernel into.
// CHECK: mogg.func @stub_1 private
// CHECK: mogg.output
mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %ctx: !mogg.context)
    attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.output
}

// CHECK-LABEL: mgp.model @dg
// The body becomes one call: the dependencies op's own builder block argument,
// then the tensor the body used from above.
// CHECK: mgp.device_graph.collective builder(%{{.*}}) stable() <"gpu", 0> {
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%[[DEPB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK-NEXT: mogg.call @[[BODY:__[A-Za-z0-9_]+]] %[[DEPB]], %{{.*}} : !mgp.device_graph.builder, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NEXT: }
// CHECK: mgp.device_graph.collective.yield
// CHECK: mogg.func @[[BODY]] private (%{{.*}}: !mgp.device_graph.builder, %{{.*}}: !mgp.tensor<2x3xf32, <"gpu", 0>>) attributes {abi = #mogg.abi<MOGG>,
// The captures re-source the body's cast of the builder and its use of the
// outer tensor.
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
// CHECK: %[[CTX:.*]] = mogg.rebind %[[CAST]] : !mgp.device_context<<"gpu", 0>> to !mogg.context
// CHECK: "test.use"(%{{.*}}) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
// CHECK: "test.tensor_out"()
// CHECK: mogg.call @stub_1
// CHECK: mogg.output
mgp.model @dg device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      "test.use"(%in) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
      %t_out = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_out, %ctx_h : !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A body that yields values is left inline: outlining it would need a stub
// return value and a re-yield at the call site.

// CHECK-LABEL: mgp.model @dg_results
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: %{{.*}} = "test.value"()
// CHECK-NEXT: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.buffer<<"gpu", 0>>
mgp.model @dg_results device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node, %r = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %v = "test.value"() : () -> !mgp.buffer<<"gpu", 0>>
      mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
    } : !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A body that is already a single call needs no wrapper: the pass is
// idempotent, so re-running it does not add a level of indirection.

// CHECK-LABEL: mogg.func @stub_1 private
// CHECK: mogg.output

// CHECK-LABEL: mgp.model @dg_outlined
// CHECK: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: mogg.call @stub_1
// CHECK-NEXT: }
mogg.func @stub_1 private dispatch(deferred) ()
    attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.output
}

mgp.model @dg_outlined device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      mogg.call @stub_1
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// Two units that record the same work merge onto one function once outlined:
// the bodies differ only in which outer tensor they read. `MergeIdenticalFuncs`
// retargets both calls, which is the deduplication the outlining exists for.

// CHECK: mogg.func @stub_1 private
// CHECK: mogg.output

// CHECK-LABEL: mgp.model @dg_dup
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: mogg.call @[[SHARED:__[A-Za-z0-9_]+]] %{{.*}}, %{{.*}} : !mgp.device_graph.builder, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%{{.*}}) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: mogg.call @[[SHARED]] %{{.*}}, %{{.*}} : !mgp.device_graph.builder, !mgp.tensor<2x3xf32, <"gpu", 0>>
mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %ctx: !mogg.context)
    attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.output
}

mgp.model @dg_dup device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node0 = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder0: !mgp.device_graph.builder):
      %ctx_cast0 = builtin.unrealized_conversion_cast %dep_builder0 : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h0 = mogg.rebind %ctx_cast0 : !mgp.device_context<<"gpu", 0>> to !mogg.context
      "test.use"(%in0) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
      %t_out0 = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_out0, %ctx_h0 : !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    %node1 = mgp.device_graph.dependencies builder(%b) dependencies(%node0) {
    ^bb0(%dep_builder1: !mgp.device_graph.builder):
      %ctx_cast1 = builtin.unrealized_conversion_cast %dep_builder1 : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h1 = mogg.rebind %ctx_cast1 : !mgp.device_context<<"gpu", 0>> to !mogg.context
      "test.use"(%in1) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
      %t_out1 = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_out1, %ctx_h1 : !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// The build region of an `mgp.device_graph.collective.create` is outlined the
// same way. The op is `IsolatedFromAbove`, so its block arguments (builder,
// shared context, inputs) are the stub's whole parameter list.

// CHECK-LABEL: mgp.model @create
// CHECK: mgp.device_graph.collective.create
// The constant stays in the region: it reads the enclosing compiled region's
// constant-resource table, which a MOGG-ABI body does not carry. It crosses as
// an argument, and so does each input the body reads; the region's own builder
// and context stay behind, since this body uses neither.
// CHECK: %[[CONST:.*]] = mgp.buffer.constant
// CHECK-NEXT: %{{.*}} = mogg.call @[[STUB:__[A-Za-z0-9_]+]] %[[CONST]], %{{.*}} : !mgp.buffer, !mgp.tensor<2x3xf32, <"gpu", 0>> -> !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NEXT: mgp.device_graph.yield
// CHECK: mogg.func @[[STUB]] private (%{{.*}}: !mgp.buffer, %{{.*}}: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> attributes {abi = #mogg.abi<MOGG>,
// CHECK: "test.use"(%{{.*}}, %{{.*}}) : (!mgp.buffer, !mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
// CHECK-NEXT: mogg.output %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
mgp.model @create device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %cache = "test.cache"() : () -> !mgp.device_graph.cache
  %graphs = mgp.device_graph.collective.create %ctx cache(%cache : !mgp.device_graph.cache) graph(%gpu : !mgp.device_context<<"gpu", 0>>) builder(%b) inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
    %c = mgp.buffer.constant #M.aligned_bytes<"0x01", align 8> : !M.array<1xui8>
    "test.use"(%c, %in) : (!mgp.buffer, !mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
    mgp.device_graph.yield %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>
  %ch, %out = mgp.device_graph.collective.execute %graphs inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>> -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A region that yields several values returns them all; the call binds one
// result per value and the yield forwards them.

// CHECK-LABEL: mgp.model @create_pair
// CHECK: %[[PAIR:.*]]:2 = mogg.call @[[STUB:__[A-Za-z0-9_]+]] %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>> -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NEXT: mgp.device_graph.yield %[[PAIR]]#0, %[[PAIR]]#1
// CHECK: mogg.func @[[STUB]] private (%{{.*}}: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>) attributes {abi = #mogg.abi<MOGG>,
// CHECK: mogg.output %{{.*}}, %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
mgp.model @create_pair device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %cache = "test.cache"() : () -> !mgp.device_graph.cache
  %graphs = mgp.device_graph.collective.create %ctx cache(%cache : !mgp.device_graph.cache) graph(%gpu : !mgp.device_context<<"gpu", 0>>) builder(%b) inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
    %a = "test.a"(%in) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %c = "test.b"(%in) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.yield %a, %c : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>)>
  %ch, %o0, %o1 = mgp.device_graph.collective.execute %graphs inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>)> -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %o0, %o1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A region that is already a single call needs no wrapper, so re-running the
// pass leaves it alone.

// CHECK-LABEL: mgp.model @create_outlined
// CHECK: mgp.device_graph.collective.create
// CHECK-NEXT: %{{.*}} = mogg.call @device_graph_build %{{.*}} : !mgp.device_graph.builder, !mgp.context, !mgp.tensor<2x3xf32, <"gpu", 0>> -> !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NEXT: mgp.device_graph.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
mogg.func @device_graph_build private (%b: !mgp.device_graph.builder, %ctx: !mgp.context, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>)
    -> !mgp.tensor<2x3xf32, <"gpu", 0>> attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"device_graph_build">>} {
  mogg.output %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
}

mgp.model @create_outlined device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %cache = "test.cache"() : () -> !mgp.device_graph.cache
  %graphs = mgp.device_graph.collective.create %ctx cache(%cache : !mgp.device_graph.cache) graph(%gpu : !mgp.device_context<<"gpu", 0>>) builder(%b) inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
    %out = mogg.call @device_graph_build %b, %ctx, %in : !mgp.device_graph.builder, !mgp.context, !mgp.tensor<2x3xf32, <"gpu", 0>> -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.yield %out : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>
  %ch, %out = mgp.device_graph.collective.execute %graphs inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>> -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A body with a single op is left alone: wrapping it would only add
// indirection.

// CHECK-LABEL: mgp.model @single_op
// CHECK: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: "test.use"()
// CHECK-NEXT: }
// CHECK-NOT: mogg.func @__device_graph_dependencies
mgp.model @single_op device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      "test.use"() : () -> ()
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A dependencies body that never records keeps its builder block argument (the
// op requires one) but does not pass it, so `producesNewNodes` still answers
// that nothing is recorded.

// CHECK-LABEL: mgp.model @unused_builder
// CHECK: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: mogg.call @[[STUB:__[A-Za-z0-9_]+]]
// CHECK-NEXT: }
// CHECK: mogg.func @[[STUB]] private () attributes {abi = #mogg.abi<MOGG>,
// CHECK: "test.use"()
// CHECK-NEXT: "test.use2"()
// CHECK-NEXT: mogg.output
mgp.model @unused_builder device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      "test.use"() : () -> ()
      "test.use2"() : () -> ()
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A chain is ordering, not data: the emitters bind no Mojo value for a
// `mogg.chain.dummy`, so the outlined body materializes its own instead of
// taking one as an argument.

// CHECK-LABEL: mgp.model @chain
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: mogg.call @[[UNIT:__[A-Za-z0-9_]+]] %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 0>>, !mgp.tensor_spec<2x3xf32>
// CHECK-NEXT: }
// The chain used to arrive as a parameter; it is created inside instead.
// CHECK: mogg.func @[[UNIT]] private (%{{.*}}: !mgp.buffer<<"gpu", 0>>, %{{.*}}: !mgp.tensor_spec<2x3xf32>) attributes {abi = #mogg.abi<MOGG>,
// CHECK: mogg.chain.dummy()
// CHECK: mgp.tensor.create %{{.*}}, %{{.*}}, %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mogg.output
mgp.model @chain device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %chain = mogg.chain.dummy()
    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %buf = mgp.tensor.extract.buffer %in : <2x3xf32, <"gpu", 0>>
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %t = mgp.tensor.create %chain, %buf, %spec : !mgp.tensor<2x3xf32, <"gpu", 0>>
      "test.use"(%t) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}
