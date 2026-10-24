// RUN: mo-opt %s --mogg-import-packages --mogg-emit-mojo --mogg-dump-mojo 2>&1 | FileCheck %s

// A `stable`-marked `!mo.tensor` lowers the same way everywhere it is
// realized: `mogg.tensor.__init__` builds it with the `StableTensor` primitive
// (an unmarked result builds a `ManagedTensorSlice` from the same pointer), and
// a kernel stub argument of that type is spelled as a `StableTensor`.

// CHECK-LABEL: def region_stable(
// CHECK: var [[SLOT:var[_0-9]+]] = builtin_primitives.primitives.unpack_buffer_ref(
// CHECK: var [[DATA:var[_0-9]+]] = builtin_primitives.primitives.unpack_buffer_ref(
// CHECK: builtin_primitives.primitives.mogg_stable_tensor_init[DType.float32, 1, False, extensibility.IO.Input, {{[0-9]+}}]([[SLOT]].unsafe_ptr().unsafe_bitcast[NoneType](), {{var[_0-9]+}})
// CHECK: builtin_primitives.primitives.mogg_tensor_init[DType.float32, 1, True, extensibility.IO.Output, {{[0-9]+}}]([[DATA]].unsafe_ptr().unsafe_bitcast[NoneType](), {{var[_0-9]+}})

// CHECK-LABEL: def consume(
// CHECK-SAME: extensibility.StableInputTensor[dtype = DType.float32, rank = 1, ...],
// CHECK-SAME: extensibility.OutputTensor[dtype = DType.float32, rank = 1, ...],

mogg.func @region_stable private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer<<"gpu", 0>>>, %arg2: !mogg.async<!mgp.buffer<<"gpu", 0>>>, %arg3: !mogg.async<!mgp.device_context<<"gpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.kernel">>} {
  %0 = mogg.async.unpack(%arg1) : !mgp.buffer<<"gpu", 0>>
  %1 = mogg.async.unpack(%arg2) : !mgp.buffer<<"gpu", 0>>
  %2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"gpu", 0>>
  %3 = mogg.tensor.__init__(%0) io = Input, mut = false : !mgp.buffer<<"gpu", 0>> -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false, stable}>
  %4 = mogg.tensor.__init__(%1) io = Output, mut = true : !mgp.buffer<<"gpu", 0>> -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @consume %3, %4, %2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false, stable}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"gpu", 0>>
  %5 = mogg.async.ready
  mogg.output %5 : !mef.chain
}

mogg.func @consume private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false, stable}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.kernel">>} {
  mogg.output
}

// The stubs need a caller, or emission's symbol DCE drops them.
mgp.model @top_level : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 1
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %input: !mgp.tensor<10xf32, <"gpu", 0>>) -> (!mef.chain) {
  %buf = mgp.tensor.extract.buffer %input : <10xf32, <"gpu", 0>>
  %ch = mogg.compiled.call @region_stable(%arg0, %buf, %buf, %arg3) fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.kernel">> constantResources = [], usedParams = [] {device = #M.device_ref<"gpu", 0>} : (!mef.chain, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  mef.output %ch : !mef.chain
}
