// RUN: mo-opt %s --split-input-file \
// RUN:     --gc-pipeline-pre-jit="kernel-library-paths=%kernel_api_test_lib dump-mojo-text=true" \
// RUN:     -o /dev/null 2>&1 | FileCheck %s

// End-to-end Mojo emission for a multi-device `mgp.device_graph.collective`.
// The collective is finalized into a `mgp.device_graph.collective.create`
// wrapped in a `mogg.routine.async`, then `mogg-emit-mojo` emits that
// region as a Mojo function.

// A single indexed build closure is generated and dispatched by device ordinal.
// CHECK: def [[BUILD:collective_build_for[_0-9]*]][n{{[_0-9]*}} : Int](mut builder: {{.*}}DeviceGraphBuilder) raises {imm}:

// Device 0 records its input as a graph-owned stable location.
// CHECK: comptime if n{{[_0-9]*}} == 0:
// CHECK: var {{[a-zA-Z_0-9]+}} = builder.add_input({{var[_0-9]+}})
// CHECK: builder.add_output({{[a-zA-Z_0-9]+}}.async_pack())

// Device 1 records its input similarly.
// CHECK: elif n{{[_0-9]*}} == 1:
// CHECK: var {{[a-zA-Z_0-9]+}} = builder.add_input({{var[_0-9]+}})
// CHECK: builder.add_output({{[a-zA-Z_0-9]+}}.async_pack())

// The cache-key callback also branches by device ordinal.
// CHECK: def [[KEY:collective_key_for[_0-9]*]][n{{[_0-9]*}} : Int]() raises {imm} -> String:
// CHECK: comptime if n{{[_0-9]*}} == 0:
// CHECK: {{.*}}DeviceGraphCache.make_key({{var[_0-9]+}})
// CHECK: elif n{{[_0-9]*}} == 1:
// CHECK: {{.*}}DeviceGraphCache.make_key({{var[_0-9]+}})
// CHECK: raise Error("unexpected device index in collective device graph key")

// The per-device contexts are assembled into an Array[DeviceContext, 2].
// CHECK: var {{[a-zA-Z_0-9]+}}: std.collections.Array[max.gpu.host.DeviceContext, 2]
// CHECK: {{[a-zA-Z_0-9]+}} = [{{var[_0-9]+}}, {{var[_0-9]+}}]

// The collective is built/reused through `DeviceGraph.create_collective`.
// CHECK: [[ARRAY:var[_0-9]+]] = max.gpu.host.DeviceGraph.create_collective({{[a-zA-Z_0-9]+}}, [[BUILD]], [[KEY]], cache = {{var[_0-9]+}})

// Each per-device graph is unpacked into its own origin-trackable variable
// before the terminator lowers the transfer.
// CHECK: var [[G0:var[_0-9]+]] = [[ARRAY]][0]
// CHECK: var [[G1:var[_0-9]+]] = [[ARRAY]][1]

// Each per-device graph result is packed back into the async ABI.
// CHECK: {{[a-zA-Z_0-9]+}} = builtin_primitives.primitives.mogg_async_pack_value([[G0]]^)
// CHECK: {{[a-zA-Z_0-9]+}} = builtin_primitives.primitives.mogg_async_pack_value([[G1]]^)

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

// A collective with a stable address slot. The slot is allocated on the
// device its type names, passed to `create_collective` as an ordinal into the
// contexts array, and reached by index through the builder in each region.
// The reading region views the slot as a stable tensor, which is an ordinary
// `OwnedTensor` over the slot's address; here it is yielded as that region's
// result.

// CHECK: def [[BUILD:collective_build_for[_0-9]*]][n{{[_0-9]*}} : Int](mut builder: {{.*}}DeviceGraphBuilder) raises {imm}:
// CHECK: comptime if n{{[_0-9]*}} == 0:
// CHECK: builder.place_stable(0, {{.*}})
// CHECK: elif n{{[_0-9]*}} == 1:
// CHECK: [[SLOT:var[_0-9]+]] = builder.get_stable(0)
// CHECK: [[SPEC:var[_0-9]+]] = builtin_primitives.primitives.mgp_tensor_spec_create[
// CHECK: builtin_primitives.primitives.mgp_device_graph_stable_tensor[2, 2, DType.float32]([[SLOT]], [[SPEC]])
// CHECK: max.gpu.host.DeviceGraph.create_collective({{[a-zA-Z_0-9]+}}, [[BUILD]], {{[a-zA-Z_0-9]+}}, cache = {{var[_0-9]+}}, stable_devices = [0])

mgp.model @dg_collective_stable device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>, stable>) {
  %c0, %c1 = mgp.device_graph.collective builder(%b) stable(
      !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    %p0 = mgp.device_graph.stable.place %b[0], %in0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.collective.yield %in0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
  }, <"gpu", 1> {
    %a0 = mgp.device_graph.stable.get %b[0] : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
    %s0 = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %t0 = mgp.device_graph.stable.tensor %a0, %s0 : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
    mgp.device_graph.collective.yield %t0 : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>, stable>

  mef.output %ch0, %c0, %c1
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
}

// -----

// gpu:1's graph reads gpu:0's tensor only for its shape, so instead of a
// stable copy it adds the tensor rebound to the `shape_only` kind: recorded in
// place, keyed on shape alone. Its own tensor still gets an ordinary
// `add_input`.
// CHECK: var [[SHAPE_ONLY:var[_0-9]+]] = {{var[_0-9]+}}.with_kind[max.gpu.host.DeviceGraphInputKind.shape_only]()
// CHECK-NEXT: builder.add_input([[SHAPE_ONLY]])
// CHECK-NEXT: builder.add_input({{var[_0-9]+}})

mgp.model @dg_collective_shape_only device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
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
