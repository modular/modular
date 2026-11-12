// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mogg-emit-mojo --mogg-dump-mojo 2>&1 | FileCheck %s
//
// The hand-written `mogg.call.build_graph` names a kernel declaration that
// only exists once `--mogg-import-packages` has run, so parse-time
// verification is disabled; every pass still verifies its output.

// A recording stub for `imposter_stable_single_build_graph`, whose
// `build_graph` takes `Some[Input & Tensor & Stable]`. The contract carries
// `stable = true` on the `mogg.trait.create` that wraps the input, and emit
// assembles a `_MoggStableTensorArg` (slot-backed, `StableAddrEngine`) instead
// of the pointer-backed `_MoggDenseTensorArg`.

// Dense and stable share one struct, `_MoggTensorArg`, parameterized on its
// engine: the `Stable` conformance and the constructor overload that wraps
// the slot both follow the engine, and `_MoggStableTensorArg` is the alias
// that pins `StableAddrEngine`.
// CHECK-DAG: struct _MoggTensorIO
// CHECK-DAG: struct _MoggTensorArg[
// CHECK-DAG: Stable where _Engine == StableAddrEngine[element_width=1],
// CHECK-DAG: ) where Self.Engine == StableAddrEngine[element_width=1]:
// CHECK-DAG: StableStorage[Self.dtype, Self.Origin, AddressSpace.GENERIC](ptr)
// CHECK-DAG: comptime _MoggStableTensorArg = _MoggTensorArg[
// CHECK-DAG: _Engine=StableAddrEngine[element_width=1],
//
// The stub assembles the stable input and the dense output, then records the
// kernel.
// CHECK-DAG: _MoggStableTensorArg[type_of({{var_+[0-9]+}}).dtype, type_of({{var_+[0-9]+}}), {{[0-9]+}}, _MoggTensorIO.INPUT]
// CHECK-DAG: _MoggDenseTensorArg[type_of({{var_+[0-9]+}}).dtype, type_of({{var_+[0-9]+}}), {{[0-9]+}}, _MoggTensorIO.OUTPUT
// CHECK-DAG: StableSingleBuildGraph.build_graph

mogg.func @consume_single_build_graph private dispatch(deferred) (%px: !mogg.pointer, %po: !mogg.pointer, %builder: !mgp.device_graph.builder) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"imposter_stable_single_build_graph">>} {
  %s = mogg.shape.create() <shape = #mosh<ape[4]> : !mosh.ape>
  %x = mogg.tensor.create(%px, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape, stable = true> : !mogg.tensor
  %out = mogg.tensor.create(%po, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %x1 = mogg.trait.create(%x) contract = <{role = input, fused = false, access = none, capabilities = none, stable = true}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  %out1 = mogg.trait.create(%out) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  mogg.call.build_graph["testing_kernels.testing_kernels.StableSingleBuildGraph"] inputs(%x1 : !mogg.tensor) outputs(%out1 : !mogg.tensor) builder(%builder) moOpName = "imposter_stable_single_build_graph"
  mogg.output
}
