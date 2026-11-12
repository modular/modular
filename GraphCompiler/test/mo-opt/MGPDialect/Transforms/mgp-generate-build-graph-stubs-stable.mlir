// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mgp-generate-build-graph-stubs --canonicalize --allow-unregistered-dialect | FileCheck %s
//
// The `mogg.call.opaque` names a kernel declaration that only exists once
// `--mogg-import-packages` has run, so parse-time verification is disabled;
// every pass still verifies its output.

// A MAP stub whose kernel takes a plain `TensorTuple` in `execute` and a
// `TensorTuple` of `Stable` elements in `build_graph`. `ConformCallOperands`
// wrapped each operand in a `mogg.trait.create` carrying `execute`'s contract;
// the recording sibling re-stamps those contracts `stable` and keeps the
// `tensor_tuple` pack ABI. The output and the original stub are left as they
// were.

// CHECK: mogg.func @stub_1 private dispatch(deferred) (
// CHECK-NOT: stable = true}>
// CHECK: mogg.call.opaque @testing_kernels.testing_kernels.StableBuildGraph

// CHECK: mogg.func @stub_1_build_graph private dispatch(deferred)
// CHECK-DAG: mogg.trait.create(%{{.*}}) contract = <{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: %[[A:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = input, fused = false, access = none, capabilities = none, stable = true}>
// CHECK-DAG: %[[B:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = input, fused = false, access = none, capabilities = none, stable = true}>
// CHECK: mogg.tuple.create(%[[A]], %[[B]]) variadic_abi = <tensor_tuple>
// CHECK: mogg.call.build_graph["testing_kernels.testing_kernels.StableBuildGraph"]
mogg.func @stub_1 private dispatch(deferred) (%p0: !mogg.pointer, %p1: !mogg.pointer, %po: !mogg.pointer, %ctx: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"imposter_stable_build_graph">>} {
  %s = mogg.shape.create() <shape = #mosh<ape[4]> : !mosh.ape>
  %a = mogg.tensor.create(%p0, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape, stable = true> : !mogg.tensor
  %b = mogg.tensor.create(%p1, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape, stable = true> : !mogg.tensor
  %out = mogg.tensor.create(%po, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %a1 = mogg.trait.create(%a) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  %b1 = mogg.trait.create(%b) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  %out1 = mogg.trait.create(%out) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  %t = mogg.tuple.create(%a1, %b1) variadic_abi = <tensor_tuple> : !mogg.tensor, !mogg.tensor
  mogg.call.opaque @testing_kernels.testing_kernels.StableBuildGraph inputs(%t : !mogg.tuple) outputs(%out1 : !mogg.tensor) context(%ctx : !mogg.context)
  mogg.output
}

// CHECK-LABEL: mgp.model @dg_trait_stable_pack
// CHECK: mogg.call @stub_1_build_graph
mgp.model @dg_trait_stable_pack device_graph devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %ctx_cast = builtin.unrealized_conversion_cast %dep_builder : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
      %ctx_h = mogg.rebind %ctx_cast : !mgp.device_context<<"gpu", 0>> to !mogg.context
      %p0 = "test.ptr"() : () -> !mogg.pointer
      %p1 = "test.ptr"() : () -> !mogg.pointer
      %po = "test.ptr"() : () -> !mogg.pointer
      mogg.call @stub_1 %p0, %p1, %po, %ctx_h : !mogg.pointer, !mogg.pointer, !mogg.pointer, !mogg.context
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}
