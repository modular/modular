// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mogg-emit-mojo --mogg-dump-mojo 2>&1 | FileCheck %s
//
// The hand-written `mogg.call.build_graph` names a kernel declaration that
// only exists once `--mogg-import-packages` has run, so parse-time
// verification is disabled; every pass still verifies its output.

// A recording stub in the shape `mgp-generate-build-graph-stubs` leaves behind
// for `imposter_stable_build_graph`: slot-backed inputs (`stable = true` on
// their `mogg.tensor.create`) packed by a `mogg.tuple.create` the pass
// re-stamped `stable` against the kernel's `build_graph` signature. Each
// stable create builds a `StableTensor` from the slot address, and the pack
// gathers those slot addresses and shapes into a `StableVariadicTensors`
// (where a plain pack would gather data pointers into `VariadicTensors`).

// CHECK-LABEL: def consume_build_graph(
// CHECK-SAME: [[P0:[a-z_0-9]+]]: OpaquePointer[MutAnyOrigin], [[P1:[a-z_0-9]+]]: OpaquePointer[MutAnyOrigin], [[PO:[a-z_0-9]+]]: OpaquePointer[MutAnyOrigin], [[BUILDER:[a-z_0-9]+]]: {{.*}}DeviceGraphBuilder) raises:
// CHECK: var [[A:var[_0-9]+]] = builtin_primitives.primitives.mogg_stable_tensor_init[DType.float32, 1, False, extensibility.IO.Input, {{[0-9]+}}]([[P0]].unsafe_bitcast[NoneType](), {{var[_0-9]+}})
// CHECK: var [[B:var[_0-9]+]] = builtin_primitives.primitives.mogg_stable_tensor_init[DType.float32, 1, False, extensibility.IO.Input, {{[0-9]+}}]([[P1]].unsafe_bitcast[NoneType](), {{var[_0-9]+}})
// CHECK: var [[OUT:var[_0-9]+]] = builtin_primitives.primitives.mogg_tensor_init[DType.float32, 1, True, extensibility.IO.Output, {{[0-9]+}}]([[PO]].unsafe_bitcast[NoneType](), {{var[_0-9]+}})
// CHECK: var [[ADDRS:var[_0-9]+]] = std.utils.StaticTuple[size = 2]([[A]]._addr, [[B]]._addr)
// CHECK: var [[SHAPES:var[_0-9]+]] = std.utils.StaticTuple[size = 2]([[A]].shape(), [[B]].shape())
// CHECK: var [[PACK:var[_0-9]+]] = extensibility.StableVariadicTensors[io_spec = [[A]].io_spec, static_specs = {{param[_0-9]+}}]([[ADDRS]], [[SHAPES]])
// CHECK: StableBuildGraph.build_graph([[OUT]], [[PACK]], [[BUILDER]])

mogg.func @consume_build_graph private dispatch(deferred) (%p0: !mogg.pointer, %p1: !mogg.pointer, %po: !mogg.pointer, %builder: !mgp.device_graph.builder) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"imposter_stable_build_graph">>} {
  %s = mogg.shape.create() <shape = #mosh<ape[4]> : !mosh.ape>
  %a = mogg.tensor.create(%p0, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape, stable = true> : !mogg.tensor
  %b = mogg.tensor.create(%p1, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape, stable = true> : !mogg.tensor
  %out = mogg.tensor.create(%po, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %t = mogg.tuple.create(%a, %b) variadic_abi = <stable> : !mogg.tensor, !mogg.tensor
  mogg.call.build_graph["testing_kernels.testing_kernels.StableBuildGraph"] inputs(%t : !mogg.tuple) outputs(%out : !mogg.tensor) builder(%builder) moOpName = "imposter_stable_build_graph"
  mogg.output
}
