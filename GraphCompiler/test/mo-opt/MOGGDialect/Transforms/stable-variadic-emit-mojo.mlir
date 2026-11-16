// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mogg-emit-mojo --mogg-dump-mojo 2>&1 | FileCheck %s
//
// The hand-written `mogg.call.build_graph` names a kernel declaration that
// only exists once `--mogg-import-packages` has run, so parse-time
// verification is disabled; every pass still verifies its output.

// A recording stub in the shape `mgp-generate-build-graph-stubs` leaves behind
// for `imposter_stable_build_graph`: each slot-backed element is a
// `mogg.trait.create` whose contract the pass re-stamped `stable`, so emit
// wraps its slot address in a `_MoggStableTensorArg`, and the `tensor_tuple`
// pack gathers those into the kernel's `TensorTuple`.

// A slot-backed `mogg.tensor.create` builds the same `ManagedTensorSlice` as
// any other, over the slot address, and the wrapper takes that pointer back off
// the slice.
// CHECK-LABEL: def consume_build_graph(
// CHECK-SAME: [[P0:[a-z_0-9]+]]: OpaquePointer[MutAnyOrigin], [[P1:[a-z_0-9]+]]: OpaquePointer[MutAnyOrigin], {{.*}}, [[BUILDER:[a-z_0-9]+]]: {{[^,]*}}DeviceGraphBuilder) raises:
// CHECK: var [[A:var[_0-9]+]] = builtin_primitives.primitives.mogg_tensor_init[DType.float32, 1, False, extensibility.IO.Input, {{[0-9]+}}]([[P0]].unsafe_bitcast[NoneType](),
// CHECK: var [[B:var[_0-9]+]] = builtin_primitives.primitives.mogg_tensor_init[DType.float32, 1, False, extensibility.IO.Input, {{[0-9]+}}]([[P1]].unsafe_bitcast[NoneType](),
// CHECK: var [[A_SLOT:var[_0-9]+]] = [[A]].unsafe_ptr().unsafe_bitcast[NoneType]().as_unsafe_any_origin()
// CHECK: var [[A_ARG:var[_0-9]+]] = _MoggStableTensorArg[{{.*}}_MoggTensorIO.INPUT]([[A_SLOT]],
// CHECK: var [[B_SLOT:var[_0-9]+]] = [[B]].unsafe_ptr().unsafe_bitcast[NoneType]().as_unsafe_any_origin()
// CHECK: var [[B_ARG:var[_0-9]+]] = _MoggStableTensorArg[{{.*}}_MoggTensorIO.INPUT]([[B_SLOT]],
// CHECK: var [[OUT:var[_0-9]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.OUTPUT
// CHECK: var [[PACK:var[_0-9]+]] = extensibility.TensorTuple[{{.*}}]([[A_ARG]], [[B_ARG]])
// CHECK: StableBuildGraph.build_graph([[OUT]], [[PACK]], [[BUILDER]])

mogg.func @consume_build_graph private dispatch(deferred) (%p0: !mogg.pointer, %p1: !mogg.pointer, %po: !mogg.pointer, %builder: !mgp.device_graph.builder) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"imposter_stable_build_graph">>} {
  %s = mogg.shape.create() <shape = #mosh<ape[4]> : !mosh.ape>
  %a = mogg.tensor.create(%p0, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %b = mogg.tensor.create(%p1, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %out = mogg.tensor.create(%po, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %a1 = mogg.trait.create(%a) contract = <{role = input, fused = false, access = none, capabilities = none, stable = true}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  %b1 = mogg.trait.create(%b) contract = <{role = input, fused = false, access = none, capabilities = none, stable = true}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  %out1 = mogg.trait.create(%out) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  %t = mogg.tuple.create(%a1, %b1) variadic_abi = <tensor_tuple> : !mogg.tensor, !mogg.tensor
  mogg.call.build_graph["testing_kernels.testing_kernels.StableBuildGraph"] inputs(%t : !mogg.tuple) outputs(%out1 : !mogg.tensor) builder(%builder) moOpName = "imposter_stable_build_graph"
  mogg.output
}
