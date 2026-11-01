// RUN: mo-opt %s --mgp-partition-device-graph-collective --allow-unregistered-dialect --split-input-file | FileCheck %s

// A multi-device `isDeviceGraph` model after `mgp-stage-device-graphs`. The
// `mgp.device_graph.staging` op wraps the per-device producers, the
// `mgp.device_graph.stable.create` ops, and the per-device consumers. This is
// the input shape expected by `mgp-partition-device-graph-collective`.
//
// Each `stable.create` is lowered per region: the producing region publishes
// its tensor with `stable.place` and takes the spec from that tensor, every
// other region reads the slot with `stable.get` and builds the spec from the
// static type, and `stable.tensor` rebuilds the `!mgp.tensor<..., stable>` the
// consumers were lowered against. The consumer regions themselves are cloned
// unchanged.
//
// CHECK-LABEL: mgp.model @multi_device_stable
// CHECK-NOT:   mgp.device_graph.staging
// CHECK:       mgp.device_graph.collective builder(%[[B:[a-z0-9_]+]]) stable(
// CHECK-SAME:      !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>,
// CHECK-SAME:      !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>)
// CHECK:       <"gpu", 0> {
// CHECK:         %[[PROD0:[a-z0-9_]+]] = mogg.routine.async
// CHECK:         %[[P0:[a-z0-9_]+]] = mgp.device_graph.stable.place %[[B]][0], %[[PROD0]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK:         %[[S0:[a-z0-9_]+]] = mgp.tensor.extract.tensor_spec %[[PROD0]] : <2x3xf32, <"gpu", 0>>
// CHECK:         %[[T0:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[P0]], %[[S0]] : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
// CHECK:         %[[G1:[a-z0-9_]+]] = mgp.device_graph.stable.get %[[B]][1] : !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>
// CHECK:         %[[S1:[a-z0-9_]+]] = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
// CHECK:         %[[T1:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[G1]], %[[S1]] : !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>
// CHECK:         mogg.routine.async (%[[T0]], %[[T1]], %{{[a-z0-9_]+}})
// CHECK-SAME:      (%[[A0:[a-z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
// CHECK-SAME:       %[[A1:[a-z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 1>, stable>,
// CHECK:           mogg.async.unpack(%[[A0]]) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
// CHECK:           mogg.async.unpack(%[[A1]]) : !mgp.tensor<2x3xf32, <"gpu", 1>, stable>
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }, <"gpu", 1> {
// CHECK:         %[[PROD1:[a-z0-9_]+]] = mogg.routine.async
// CHECK:         %[[P1:[a-z0-9_]+]] = mgp.device_graph.stable.place %[[B]][1], %[[PROD1]]
// CHECK:         %[[S1B:[a-z0-9_]+]] = mgp.tensor.extract.tensor_spec %[[PROD1]]
// CHECK:         %[[T1B:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[P1]], %[[S1B]]
// CHECK:         %[[G0:[a-z0-9_]+]] = mgp.device_graph.stable.get %[[B]][0]
// CHECK:         %[[S0B:[a-z0-9_]+]] = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
// CHECK:         %[[T0B:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[G0]], %[[S0B]]
// CHECK:         mogg.routine.async (%[[T0B]], %[[T1B]], %{{[a-z0-9_]+}})
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }

mgp.model @multi_device_stable device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) {

  %c0, %c1 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %in1, %gpu1
          : !mgp.tensor<2x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.tensor<2x3xf32, <"gpu", 1>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<2x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %in1_t: !mgp.tensor<2x3xf32, <"gpu", 1>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):

    // Producer on gpu:0.
    %a = mogg.routine.async (%in0_t, %gpu0_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}

    // Producer on gpu:1.
    %b = mogg.routine.async (%in1_t, %gpu1_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 1>>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}

    // Promote each producer result to a stable address inside the staging body.
    %s0 = mgp.device_graph.stable.create %a : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %s1 = mgp.device_graph.stable.create %b : !mgp.tensor<2x3xf32, <"gpu", 1>>

    // Consumer on gpu:0, reading both stable values.
    // The stable flag is an internal device-graph detail: the value returned
    // from the staging region escapes as a normal tensor.
    %c0 = mogg.routine.async (%s0, %s1, %gpu0_t)
        : (%arg0: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
           %arg1: !mgp.tensor<2x3xf32, <"gpu", 1>, stable>,
            %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
      %u1 = mogg.async.unpack(%arg1) : !mgp.tensor<2x3xf32, <"gpu", 1>, stable>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %v = builtin.unrealized_conversion_cast %u0
          : !mgp.tensor<2x3xf32, <"gpu", 0>, stable> to !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%v) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}

    // Consumer on gpu:1, reading both stable values.
    %c1 = mogg.routine.async (%s0, %s1, %gpu1_t)
        : (%arg0: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
           %arg1: !mgp.tensor<2x3xf32, <"gpu", 1>, stable>,
            %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
      %u1 = mogg.async.unpack(%arg1) : !mgp.tensor<2x3xf32, <"gpu", 1>, stable>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %v = builtin.unrealized_conversion_cast %u1
          : !mgp.tensor<2x3xf32, <"gpu", 1>, stable> to !mgp.tensor<2x3xf32, <"gpu", 1>>
      %p = mogg.async.pack(%v) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}

    mgp.device_graph.staging.yield %c0, %c1
        : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0, %c1
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
}

// -----

// A consumer lives on a device that has no producer of its own. The partition
// must still create a region for that device so the consumer can read the
// remote stable value and yield its result.
//
// CHECK-LABEL: mgp.model @consumer_only_device
// CHECK-NOT:   mgp.device_graph.staging
// CHECK:       mgp.device_graph.collective builder(%[[B:[a-z0-9_]+]]) stable(
// CHECK-SAME:      !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>)
// CHECK:       <"gpu", 0> {
// CHECK:         mogg.routine.async
// CHECK:         mgp.device_graph.stable.place
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }, <"gpu", 1> {
// CHECK:         %[[G0:[a-z0-9_]+]] = mgp.device_graph.stable.get %[[B]][0]
// CHECK:         %[[S0:[a-z0-9_]+]] = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
// CHECK:         %[[T0:[a-z0-9_]+]] = mgp.device_graph.stable.tensor %[[G0]], %[[S0]]
// CHECK:         mogg.routine.async (%[[T0]], %{{[a-z0-9_]+}})
// CHECK-SAME:      (%[[A0:[a-z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
// CHECK:           mogg.async.unpack(%[[A0]]) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
// CHECK:         mgp.device_graph.collective.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 1>>
// CHECK:       } : !mgp.tensor<2x3xf32, <"gpu", 1>>

mgp.model @consumer_only_device device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>) {

  %c0 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %gpu1
          : !mgp.tensor<2x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<2x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):

    // Producer on gpu:0.
    %a = mogg.routine.async (%in0_t, %gpu0_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}

    // Promote the producer result to a stable address.
    %s0 = mgp.device_graph.stable.create %a : !mgp.tensor<2x3xf32, <"gpu", 0>>

    // Consumer on gpu:1, reading the stable value from gpu:0.
    %c0 = mogg.routine.async (%s0, %gpu1_t)
        : (%arg0: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %v = builtin.unrealized_conversion_cast %u0
          : !mgp.tensor<2x3xf32, <"gpu", 0>, stable> to !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%v) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}

    mgp.device_graph.staging.yield %c0
        : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>
}

// -----

// Host bookkeeping goes where its operands are. The wrapper over gpu:1's
// producer result is cloned into the gpu:1 region only; cloning it into gpu:0
// as well would leave its operand pointing at a value that region never
// materializes.
//
// CHECK-LABEL: mgp.model @host_wrapper_follows_its_device
// CHECK:       <"gpu", 0> {
// CHECK:         fusionInfo = #producer0
// CHECK-NOT:     #host_wrap
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }, <"gpu", 1> {
// CHECK:         %[[PROD1:[a-z0-9_]+]] = mogg.routine.async
// CHECK:         fusionInfo = #producer1
// CHECK:         mogg.routine.async (%[[PROD1]])
// CHECK:         fusionInfo = #host_wrap
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }

mgp.model @host_wrapper_follows_its_device device_graph : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  mef.output %ch0, %ctx, %weights, %cpu, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %cpu: !mgp.device_context<<"cpu", 0>>,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) {

  %c0, %c1 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %in1, %gpu1
          : !mgp.tensor<2x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.tensor<2x3xf32, <"gpu", 1>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<2x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %in1_t: !mgp.tensor<2x3xf32, <"gpu", 1>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):

    %a = mogg.routine.async (%in0_t, %gpu0_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}

    %b = mogg.routine.async (%in1_t, %gpu1_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 1>>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}

    // Host-side bookkeeping over gpu:1's result.
    %w = mogg.routine.async (%b)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.host_wrap">> inOutSignature = <""> {device = #M.device_ref<"cpu", 0>}

    mgp.device_graph.staging.yield %a, %w
        : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0, %c1
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
}

// -----

// A stable tensor with a dynamic dimension crosses devices: the consumer's
// region rebuilds the spec from the create's runtime dim, which is a staging
// input (derived from a graph input) and so is visible in every region.
//
// CHECK-LABEL: mgp.model @dynamic_stable_crosses_devices
// CHECK:       %[[N:[a-z0-9_]+]] = mgp.tensor_spec.get_dim[0]
// CHECK:       mgp.device_graph.collective builder(%[[B:[a-z0-9_]+]]) stable(
// CHECK-SAME:      !mgp.stable_addr<<?x3xf32, <"gpu", 0>>>)
// CHECK:       <"gpu", 0> {
// CHECK:         mgp.device_graph.stable.place %[[B]][0]
// CHECK:       }, <"gpu", 1> {
// CHECK:         %[[G0:[a-z0-9_]+]] = mgp.device_graph.stable.get %[[B]][0]
// CHECK:         %[[S0:[a-z0-9_]+]] = mgp.tensor_spec.create %[[N]] : !mgp.tensor_spec<?x3xf32>
// CHECK:         mgp.device_graph.stable.tensor %[[G0]], %[[S0]]
// CHECK:         mgp.device_graph.collective.yield

mgp.model @dynamic_stable_crosses_devices device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<?x3xf32, <"gpu", 0>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>) {
  %spec = mgp.tensor.extract.tensor_spec %in0 : <?x3xf32, <"gpu", 0>>
  %n = mgp.tensor_spec.get_dim[0] %spec : !mgp.tensor_spec<?x3xf32>
  %c1 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %gpu1, %n
          : !mgp.tensor<?x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.device_context<<"gpu", 1>>,
            index) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<?x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>,
         %n_t: index):
    %a = mogg.routine.async (%in0_t, %gpu0_t)
        : (%arg: !mgp.tensor<?x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<?x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<?x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<?x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<?x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    %s0 = mgp.device_graph.stable.create %a[%n_t] : !mgp.tensor<?x3xf32, <"gpu", 0>>
    %c1 = mogg.routine.async (%s0, %gpu1_t)
        : (%arg0: !mgp.tensor<?x3xf32, <"gpu", 0>, stable>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<?x3xf32, <"gpu", 0>, stable>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      %r = "test.kernel"(%u0, %c)
          : (!mgp.tensor<?x3xf32, <"gpu", 0>, stable>,
             !mgp.device_context<<"gpu", 1>>) -> !mgp.tensor<2x3xf32, <"gpu", 1>>
      %p = mogg.async.pack(%r) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}
    mgp.device_graph.staging.yield %c1 : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 1>>
  mef.output %ch0, %c1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>
}

// -----

// A host value yielded from the staging body is cloned into, and yielded
// from, every device region, so the collective carries one result per region
// for it. Results after the duplicate must still map to the right index:
// the gpu:1 producer is the collective's fourth result, not its third.
//
// CHECK-LABEL: mgp.model @host_yield_in_every_region
// CHECK:       %[[C:[a-z0-9_]+]]:4 = mgp.device_graph.collective
// CHECK:       <"gpu", 0> {
// CHECK:         mgp.device_graph.collective.yield %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK:       }, <"gpu", 1> {
// CHECK:         mgp.device_graph.collective.yield %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
// CHECK:       }
// CHECK:       mef.output %{{[a-z0-9_]+}}, %[[C]]#0, %[[C]]#1, %[[C]]#3

mgp.model @host_yield_in_every_region device_graph : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  mef.output %ch0, %ctx, %weights, %cpu, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %cpu: !mgp.device_context<<"cpu", 0>>,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>,
           %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>)
    -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) {

  %h, %c0, %c1 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %in1, %gpu1
          : !mgp.tensor<2x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.tensor<2x3xf32, <"gpu", 1>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<2x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %in1_t: !mgp.tensor<2x3xf32, <"gpu", 1>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):

    // Host bookkeeping over a staging input: lands in both regions.
    %w = mogg.routine.async (%in0_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.host_wrap">> inOutSignature = <""> {device = #M.device_ref<"cpu", 0>}

    %a = mogg.routine.async (%in0_t, %gpu0_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 0>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}

    %b = mogg.routine.async (%in1_t, %gpu1_t)
        : (%arg: !mgp.tensor<2x3xf32, <"gpu", 1>>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}

    mgp.device_graph.staging.yield %w, %a, %b
        : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %h, %c0, %c1
      : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
}

// -----

// A kernel ordered after another device's kernel through a chain. Device
// graphs have no chains and the next pass derives a graph's edges from every
// operand, so the remote chain carries nothing this region can express: it is
// replaced by a placeholder chain, while the own-device chain is kept. A
// collective waits for its peers through the signal buffers instead.
// CHECK-LABEL: mgp.model @remote_chain_is_dropped
// CHECK:       mgp.device_graph.collective
// CHECK:       <"gpu", 0> {
// CHECK:         %[[PROD0:[a-z0-9_]+]] = mogg.routine.async
// CHECK:         %[[DUMMY:[a-z0-9_]+]] = mogg.chain.dummy
// CHECK:         mogg.routine.async (%[[PROD0]], %[[DUMMY]], %{{[a-z0-9_]+}})
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }, <"gpu", 1> {
// CHECK:         mogg.routine.async
// CHECK-NOT:     mogg.chain.dummy
// CHECK:         mgp.device_graph.collective.yield
// CHECK:       }
mgp.model @remote_chain_is_dropped device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1
      : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">,
        !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">,
           %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>)
    -> (!mef.chain, !mef.chain) {
  %out = mgp.device_graph.staging
      inputs(%gpu0, %gpu1
          : !mgp.device_context<<"gpu", 0>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):
    %a = mogg.routine.async (%gpu0_t)
        : (%devctx: !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %r = mogg.async.ready
      mogg.output %r : !mef.chain
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    %b = mogg.routine.async (%gpu1_t)
        : (%devctx: !mgp.device_context<<"gpu", 1>>) -> !mef.chain {
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %r = mogg.async.ready
      mogg.output %r : !mef.chain
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}
    // Ordered after both producers; only the gpu:0 one exists in its region.
    %c0 = mogg.routine.async (%a, %b, %gpu0_t)
        : (%own: !mef.chain, %remote: !mef.chain,
           %devctx: !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%c) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %r = mogg.async.ready
      mogg.output %r : !mef.chain
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    mgp.device_graph.staging.yield %c0 : !mef.chain
  } -> !mef.chain
  mef.output %ch0, %out : !mef.chain, !mef.chain
}
