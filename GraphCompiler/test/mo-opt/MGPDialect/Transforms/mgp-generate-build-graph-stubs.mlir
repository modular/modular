// RUN: mo-opt %s --mgp-generate-build-graph-stubs --canonicalize --allow-unregistered-dialect --split-input-file | FileCheck %s

// `--canonicalize` runs after the pass so the builder->device-context->builder
// round-trip (this pass bridges the launch operand to a builder; Stage 2 left
// the inverse cast) folds away, leaving the recording call threading the
// dependencies builder block arg directly -- which is what we match below.

// Stage 4 of device-graph synthesis. Input IR mirrors the shape produced by
// `mgp-compute-device-graph-dependencies`: the execute block holds an
// `mgp.device_graph.collective` op whose per-device region contains
// `mgp.device_graph.dependencies` ops, and the kernel-dispatch node invokes a
// module-scope `mogg.func @stub_1 private` via `mogg.call`. The kernel
// decl, stub body, and model are minimal synthetic stand-ins (the real builtin
// kernels lack a `build_graph` method, which `mogg.call.build_graph`
// verification requires).
//
// For each staged `mogg.call @stub_N` the pass generates a sibling recording
// stub `@stub_N_build_graph` (producer dispatch `mogg.call.opaque` -> recording
// `mogg.call.build_graph` against a `!mgp.device_graph.builder`) and retargets
// the call to it, threading the dependencies op's builder block arg as the
// launch operand.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

// The original stub is retained (still used by the direct-execution / init path),
// keeping its `!mogg.context` launch arg and `mogg.call.opaque` dispatch.
// CHECK: mogg.func @stub_1 private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.tensor, %{{.*}}: !mogg.tensor, %{{.*}}: !mogg.context)
// CHECK: mogg.call.opaque @k
//
// The recording sibling is generated immediately after the original stub (so it
// precedes the model): builder launch arg, void result (node-free), producer
// recorded via `mogg.call.build_graph`, and the producer's `moOpName` carried
// over.
// CHECK: mogg.func @stub_1_build_graph private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.tensor, %{{.*}}: !mogg.tensor, %[[B:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder)
// CHECK-NEXT: mogg.call.build_graph["k"] inputs(%{{.*}}) outputs(%{{.*}}) builder(%[[B]]) {{.*}}moOpName = "test.k"
// CHECK-NEXT: mogg.output
// CHECK-NEXT: }
mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.call.opaque @k inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  mogg.output
}

// CHECK-LABEL: mgp.model @dg
// The staged call is retargeted to the sibling. The launch operand is the
// *enclosing* `mgp.device_graph.dependencies` op's builder block arg (not the
// collective builder), and the now-dead builder->device-context bridge cast
// that `mgp-compute-device-graph-dependencies` left feeding the call is removed.
// CHECK: mgp.device_graph.collective builder(%{{.*}}) stable() <"gpu", 0> {
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%[[DEPB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK-NOT: unrealized_conversion_cast
// CHECK: mogg.call @stub_1_build_graph %{{.*}}, %{{.*}}, %[[DEPB]] : {{.*}}, !mgp.device_graph.builder
// CHECK-NOT: -> !mgp.device_graph.node
mgp.model @dg device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      // The launch context is sourced from the dependencies builder, exactly as
      // `mgp-compute-device-graph-dependencies` emits it.
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %t_in = "test.tensor_in"() : () -> !mogg.tensor
      %t_out = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_in, %t_out, %ctx_h : !mogg.tensor, !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A model WITHOUT `isDeviceGraph` is left untouched: no sibling stub is generated
// and the staged call still targets `@stub_1`.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

// CHECK-NOT: @stub_1_build_graph
mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.call.opaque @k inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  mogg.output
}

// CHECK-LABEL: mgp.model @not_dg
// The model is untouched: the bridge cast survives and the call still targets
// `@stub_1` with its rebound launch operand.
// CHECK: builtin.unrealized_conversion_cast
// CHECK: mogg.call @stub_1 %{{.*}}, %{{.*}}, %{{.*}} : !mogg.tensor, !mogg.tensor, !mogg.context
mgp.model @not_dg : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %node = mgp.device_graph.dependencies builder(%builder) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %t_in = "test.tensor_in"() : () -> !mogg.tensor
      %t_out = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_in, %t_out, %ctx_h : !mogg.tensor, !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.staging.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A device-graph model whose kernel has only an `execute` method (no
// `build_graph`). The recording sibling keeps the `mogg.call.opaque` producer
// but repoints the launch context to a recording device context viewed off the
// builder via `mgp.device_graph.recording_context`, so the ordinary producer
// records into the graph. No `mogg.call.build_graph` is generated.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf struct_params = [] is_user_defined = false

// CHECK: mogg.func @stub_1_build_graph private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.tensor, %{{.*}}: !mogg.tensor, %[[B:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder)
// CHECK-NEXT: %[[REC:[a-zA-Z0-9_]+]] = mgp.device_graph.recording_context %[[B]] : !mogg.context
// CHECK-NOT: mogg.call.build_graph
// CHECK: mogg.call.opaque @k inputs(%{{.*}}) outputs(%{{.*}}) context(%[[REC]]
mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.call.opaque @k inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  mogg.output
}

// CHECK-LABEL: mgp.model @dg_execute_only
// The staged call is retargeted to the recording sibling, threading the
// enclosing dependencies builder block arg as the launch operand.
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK: mogg.call @stub_1_build_graph %{{.*}}, %{{.*}}, %{{.*}} : {{.*}}, !mgp.device_graph.builder
mgp.model @dg_execute_only device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %t_in = "test.tensor_in"() : () -> !mogg.tensor
      %t_out = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_in, %t_out, %ctx_h : !mogg.tensor, !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// `mgp.buffer.alloc` in the collective body: host and device allocations alike
// are routed through the builder (`mgp.device_graph.alloc`), which allocates
// from the graph's memory pool so every recorded address stays reserved. The
// allocation's device survives in the result buffer type, which is what selects
// the host or device pool at emission time.

// CHECK-LABEL: mgp.model @dg_host_alloc
// Both allocations take the collective body's builder directly -- an alloc
// outside any `dependencies` region has no enclosing region builder to source.
// CHECK: mgp.device_graph.collective builder(%[[B:[a-zA-Z0-9_]+]]) stable() <"gpu", 0> {
// CHECK-NOT: mgp.buffer.alloc
// A `cpu` buffer type prints without its device, so the host alloc's result
// type is a bare `!mgp.buffer`.
// CHECK: %[[HOST:[a-zA-Z0-9_]+]] = mgp.device_graph.alloc %{{.*}}, %{{.*}}, %[[B]], align 16 : !mgp.buffer{{$}}
// CHECK-NOT: mgp.buffer.alloc
// CHECK: %[[DEV:[a-zA-Z0-9_]+]] = mgp.device_graph.alloc %{{.*}}, %{{.*}}, %[[B]], align 16 : !mgp.buffer<<"gpu", 0>>
// CHECK: "test.sink"(%[[HOST]], %[[DEV]])
mgp.model @dg_host_alloc device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %chain = "test.chain"() : () -> !mef.chain
    %size = "test.size"() : () -> index
    %cpu_ctx = "test.cpu_ctx"() : () -> !mgp.device_context<<"cpu", 0>>
    %gpu_ctx = "test.gpu_ctx"() : () -> !mgp.device_context<<"gpu", 0>>
    // Host metadata allocation: served from the pool's host memory manager.
    %host_buf = mgp.buffer.alloc %chain, %size, %cpu_ctx, align 16 : !mgp.buffer<<"cpu", 0>>, !mgp.device_context<<"cpu", 0>>
    %dev_buf = mgp.buffer.alloc %chain, %size, %gpu_ctx, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    "test.sink"(%host_buf, %dev_buf) : (!mgp.buffer<<"cpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> ()
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// An allocation nested in a `mgp.device_graph.dependencies` body takes *that*
// body's builder block argument, keeping the body self-contained.

// CHECK-LABEL: mgp.model @dg_alloc_in_dependencies
// CHECK: mgp.device_graph.collective builder(%{{.*}}) stable() <"gpu", 0> {
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%[[DB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK: %[[BUF:[a-zA-Z0-9_]+]] = mgp.device_graph.alloc %{{.*}}, %{{.*}}, %[[DB]], align 16 : !mgp.buffer{{$}}
// CHECK: "test.sink"(%[[BUF]])
mgp.model @dg_alloc_in_dependencies device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %chain = "test.chain"() : () -> !mef.chain
      %size = "test.size"() : () -> index
      %cpu_ctx = "test.cpu_ctx"() : () -> !mgp.device_context<<"cpu", 0>>
      %host_buf = mgp.buffer.alloc %chain, %size, %cpu_ctx, align 16 : !mgp.buffer<<"cpu", 0>>, !mgp.device_context<<"cpu", 0>>
      "test.sink"(%host_buf) : (!mgp.buffer<<"cpu", 0>>) -> ()
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// The launch context need not be the stub's last argument: here it sits
// between the input and output tensors. The sibling retypes that same slot to
// `!mgp.device_graph.builder`, and the retargeted call threads the enclosing
// dependencies builder in that position.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

// CHECK: mogg.func @stub_1_build_graph private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.tensor, %{{.*}}: !mgp.device_graph.builder, %{{.*}}: !mogg.tensor)
// CHECK-NEXT: mogg.call.build_graph["k"]
mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %ctx: !mogg.context, %out: !mogg.tensor) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.call.opaque @k inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  mogg.output
}

// CHECK-LABEL: mgp.model @dg_ctx_mid
// CHECK: mgp.device_graph.collective builder(%{{.*}}) stable() <"gpu", 0> {
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%[[DEPB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK-NOT: unrealized_conversion_cast
// CHECK: mogg.call @stub_1_build_graph %{{.*}}, %[[DEPB]], %{{.*}} : !mogg.tensor, !mgp.device_graph.builder, !mogg.tensor
mgp.model @dg_ctx_mid device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %t_in = "test.tensor_in"() : () -> !mogg.tensor
      %t_out = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_in, %ctx_h, %t_out : !mogg.tensor, !mogg.context, !mogg.tensor
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A stub with a second context-typed argument that is NOT the producer's
// launch context. Only the producer's slot becomes a builder: the other
// context argument keeps its `!mogg.context` type in the sibling, and the
// retargeted call passes its operand through unbridged (bridging every
// context-typed operand would fail verification against the sibling).

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf struct_params = [] is_user_defined = false

// CHECK: mogg.func @stub_1_build_graph private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.context, %{{.*}}: !mogg.tensor, %{{.*}}: !mogg.tensor, %[[B:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder)
// CHECK-NEXT: %[[REC:[a-zA-Z0-9_]+]] = mgp.device_graph.recording_context %[[B]] : !mogg.context
// CHECK: mogg.call.opaque @k inputs(%{{.*}}) outputs(%{{.*}}) context(%[[REC]]
mogg.func @stub_1 private dispatch(deferred) (%aux: !mogg.context, %in: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.call.opaque @k inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  mogg.output
}

// CHECK-LABEL: mgp.model @dg_two_ctx
// CHECK: mgp.device_graph.collective builder(%{{.*}}) stable() <"gpu", 0> {
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = "test.aux_ctx"()
// CHECK: mogg.call @stub_1_build_graph %[[AUX]], %{{.*}}, %{{.*}}, %{{.*}} : !mgp.device_context<<"gpu", 0>>, !mogg.tensor, !mogg.tensor, !mgp.device_graph.builder
mgp.model @dg_two_ctx device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %aux_ctx = "test.aux_ctx"() : () -> !mgp.device_context<<"gpu", 0>>
      %t_in = "test.tensor_in"() : () -> !mogg.tensor
      %t_out = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %aux_ctx, %t_in, %t_out, %ctx_h : !mgp.device_context<<"gpu", 0>>, !mogg.tensor, !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A MAP-path stub. The producer is `mogg.call.foreach`, which implements no
// `KernelTerminator` and sits in no `mogg.with_trace`, and the call site reaches
// the launch context through `mogg.rebind` -- the MAP ABI's handle bridge. The
// recording sibling repoints the producer at a recording context; the call takes
// the dependencies builder directly, leaving the `_rebind` dead.

mogg.primitive.decl @foreach_impl registration = "mogg.call.foreach" function = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

// CHECK: mogg.func @stub_1_build_graph private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.pointer, %[[B:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder)
// CHECK-NEXT: %[[REC:[a-zA-Z0-9_]+]] = mgp.device_graph.recording_context %[[B]] : !mogg.context
// CHECK-NOT: mogg.call.build_graph
// CHECK: mogg.call.foreach @foreach_impl shape(%{{.*}}) context(%[[REC]])
mogg.func @stub_1 private dispatch(deferred) (%p: !mogg.pointer, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  %shape = "test.shape"() : () -> !mogg.shape
  %t = "test.tensor"() : () -> !mogg.tensor
  mogg.call.foreach @foreach_impl shape(%shape) context(%ctx) (%i : !map.index<2>) capturing {%a = %t : !mogg.tensor} device = "gpu" {
  }
  mogg.output
}

// CHECK-LABEL: mgp.model @dg_map_foreach
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%[[DB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK-NEXT: mogg.call @stub_1_build_graph %{{.*}}, %[[DB]] : !mogg.pointer, !mgp.device_graph.builder
// CHECK-NOT: mogg.rebind
mgp.model @dg_map_foreach device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %p = "test.pointer"() : () -> !mogg.pointer
      mogg.call @stub_1 %p, %ctx_h : !mogg.pointer, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// The other MAP producer: `mogg.call.opaque` whose kernel declares
// `build_graph`. Its dispatch is replaced by `mogg.call.build_graph` and, with
// no `mogg.with_trace` wrapping it, has to be erased on its own.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

// CHECK: mogg.func @stub_1_build_graph private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.tensor, %{{.*}}: !mogg.tensor, %[[B:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder)
// CHECK: mogg.call.build_graph["k"]
// CHECK-NOT: mogg.call.opaque
mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.call.opaque @k inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  mogg.output
}

// CHECK-LABEL: mgp.model @dg_map_opaque
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%[[DB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK-NEXT: mogg.call @stub_1_build_graph %{{.*}}, %{{.*}}, %[[DB]] : {{.*}}!mgp.device_graph.builder
mgp.model @dg_map_opaque device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %t_in = "test.tensor_in"() : () -> !mogg.tensor
      %t_out = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_in, %t_out, %ctx_h : !mogg.tensor, !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A multi-device `isDeviceGraph` model whose execute body is a
// `mgp.device_graph.collective`. Each device region contains a single
// `mgp.device_graph.dependencies` with a `mogg.call`. The pass generates
// one recording sibling and retargets both device regions to it, threading the
// per-region `dependencies` builder through directly; the now-dead
// builder->device-context bridge is canonicalized away.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

// The recording sibling is generated once and shared across both collective
// regions.
// CHECK: mogg.func @stub_1 private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.tensor, %{{.*}}: !mogg.tensor, %{{.*}}: !mogg.context)
// CHECK: mogg.func @stub_1_build_graph private {{(dispatch\([a-z]+\) )?}}(%{{.*}}: !mogg.tensor, %{{.*}}: !mogg.tensor, %[[B:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder)
// CHECK-NEXT: mogg.call.build_graph["k"] inputs(%{{.*}}) outputs(%{{.*}}) builder(%[[B]]) {{.*}}moOpName = "test.k"
mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.call.opaque @k inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  mogg.output
}

// CHECK-LABEL: mgp.model @dg_collective
// CHECK: mgp.device_graph.collective builder(%[[CB:[a-zA-Z0-9_]+]]) stable() <"gpu", 0> {
// CHECK: mgp.device_graph.dependencies builder(%[[CB]]) dependencies() {
// CHECK-NEXT: ^bb0(%[[DEPB0:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: mogg.call @stub_1_build_graph %{{.*}}, %{{.*}}, %[[DEPB0]] : {{.*}}, !mgp.device_graph.builder
// CHECK: mgp.device_graph.collective.yield
// CHECK: }, <"gpu", 1> {
// CHECK: mgp.device_graph.dependencies builder(%[[CB]]) dependencies() {
// CHECK-NEXT: ^bb0(%[[DEPB1:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: mogg.call @stub_1_build_graph %{{.*}}, %{{.*}}, %[[DEPB1]] : {{.*}}, !mgp.device_graph.builder
// CHECK: mgp.device_graph.collective.yield
// CHECK: }
mgp.model @dg_collective device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>] init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node0 = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder0: !mgp.device_graph.builder):
      %ctx_cast0 = builtin.unrealized_conversion_cast %dep_builder0 : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h0 = mogg.rebind %ctx_cast0 : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %t_in0 = "test.tensor_in"() : () -> !mogg.tensor
      %t_out0 = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_in0, %t_out0, %ctx_h0 : !mogg.tensor, !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }, <"gpu", 1> {
    %node1 = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder1: !mgp.device_graph.builder):
      %ctx_cast1 = builtin.unrealized_conversion_cast %dep_builder1 : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 1>>
      %ctx_h1 = mogg.rebind %ctx_cast1 : !mgp.device_context<<"gpu", 1>> to !mogg.context
      %t_in1 = "test.tensor_in"() : () -> !mogg.tensor
      %t_out1 = "test.tensor_out"() : () -> !mogg.tensor
      mogg.call @stub_1 %t_in1, %t_out1, %ctx_h1 : !mogg.tensor, !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A fused dispatch records through `build_graph` with the fusion flags copied
// over, so emission sets the kernel's `has_epilogue_fusion` the way it does
// for the `execute` call.

#kf = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

mogg.kernel.decl @k registration = "test.k" device_info = <"", "", "", ""> execute = #kf build_graph = #kf struct_params = [] is_user_defined = false

// CHECK: mogg.func @stub_1_build_graph
// CHECK-NEXT: mogg.call.build_graph["k"] inputs(%{{.*}}) outputs(%{{.*}}) builder(%{{.*}}) {{.*}}moOpName = "test.k" {{.*}}hasEpilogueFusion = true
mogg.func @stub_1 private dispatch(deferred) (%in: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  mogg.call.opaque @k inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context) hasEpilogueFusion = true
  mogg.output
}

// CHECK-LABEL: mgp.model @dg_fused
// CHECK: mogg.call @stub_1_build_graph
mgp.model @dg_fused device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %in = "test.tensor"() : () -> !mogg.tensor
      %out = "test.tensor"() : () -> !mogg.tensor
      mogg.call @stub_1 %in, %out, %ctx_h : !mogg.tensor, !mogg.tensor, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}
