// RUN: mo-opt %s --mgp-partition-device-graph-collective --allow-unregistered-dialect --split-input-file --verify-diagnostics

// A stable slot carries only a data pointer, so a consumer on another device
// rebuilds the shape from the create's runtime dims. A dim computed by a
// kernel on the producing device never exists in the consumer's region.

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

  %c0 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %gpu1
          : !mgp.tensor<?x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<?x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
         %gpu1_t: !mgp.device_context<<"gpu", 1>>):

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

    // A data-dependent dimension: computed on gpu:0, so it only exists there.
    %n = mogg.routine.async (%gpu0_t)
        : (%devctx: !mgp.device_context<<"gpu", 0>>) -> index {
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 0>>
      %d = "test.dim"(%c) : (!mgp.device_context<<"gpu", 0>>) -> index
      %p = mogg.async.pack(%d) : index
      mogg.output %p : !mogg.async<index>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.dim0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    // expected-error@+1 {{'mgp.device_graph.stable.create' op runtime dimension 0 is produced in another device's region; a slot carries only the data pointer, so a consumer's shape must come from values every region can see}}
    %s0 = mgp.device_graph.stable.create %a[%n] : !mgp.tensor<?x3xf32, <"gpu", 0>>

    %c0 = mogg.routine.async (%s0, %gpu1_t)
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

    mgp.device_graph.staging.yield %c0 : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 1>>

  mef.output %ch0, %c0 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>
}

// -----

// A device kernel that reads another device's result directly, with no stable
// slot in between, cannot be placed: the value never exists in its region.
mgp.model @device_reads_remote_without_slot device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
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
  %c1 = mgp.device_graph.staging
      inputs(%in0, %gpu0, %gpu1
          : !mgp.tensor<2x3xf32, <"gpu", 0>>,
            !mgp.device_context<<"gpu", 0>>,
            !mgp.device_context<<"gpu", 1>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in0_t: !mgp.tensor<2x3xf32, <"gpu", 0>>,
         %gpu0_t: !mgp.device_context<<"gpu", 0>>,
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
    // expected-error@+1 {{'mogg.routine.async' op operand 0 is produced in another device's region and does not go through a stable address slot}}
    %c1 = mogg.routine.async (%a, %gpu1_t)
        : (%arg0: !mgp.tensor<2x3xf32, <"gpu", 0>>,
           %devctx: !mgp.device_context<<"gpu", 1>>)
        -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %c = mogg.async.unpack(%devctx) : !mgp.device_context<<"gpu", 1>>
      %r = "test.kernel"(%u0, %c)
          : (!mgp.tensor<2x3xf32, <"gpu", 0>>,
             !mgp.device_context<<"gpu", 1>>) -> !mgp.tensor<2x3xf32, <"gpu", 1>>
      %p = mogg.async.pack(%r) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer1">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}
    mgp.device_graph.staging.yield %c1 : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 1>>
  mef.output %ch0, %c1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 1>>
}
