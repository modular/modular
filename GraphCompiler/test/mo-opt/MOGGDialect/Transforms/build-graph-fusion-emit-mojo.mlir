// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mogg-emit-mojo --mogg-dump-mojo 2>&1 | FileCheck %s
//
// The hand-written `mogg.call.build_graph` names a kernel declaration that
// only exists once `--mogg-import-packages` has run, so parse-time
// verification is disabled; every pass still verifies its output.

// A recording stub whose dispatch carried an epilogue fusion: the call's
// `hasEpilogueFusion` flag sets the kernel's `has_epilogue_fusion` parameter,
// as `mogg.call.opaque` does for the `execute` call. The output is the
// trait-lane fused store MAP lowering produces, a `mogg.tensor.create.epilogue`
// carrying the kernel's `Output & Fused & StoreAccess` contract, so emit hands
// `build_graph` its store-fusion wrapper.

// CHECK-LABEL: def record_fused(
// CHECK-SAME: [[BUILDER:[a-z_0-9]+]]: {{[^,]*}}DeviceGraphBuilder) raises:
// CHECK: var [[OUT:var[_0-9]+]] = _MoggOutputStoreFusionTensor[DType.float32,
// CHECK: var [[X:var[_0-9]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.INPUT
// CHECK: FusedOutputBuildGraph.build_graph[has_epilogue_fusion = True]([[OUT]], [[X]], [[BUILDER]])

mogg.func @record_fused private dispatch(deferred) (%px: !mogg.pointer, %po: !mogg.pointer, %builder: !mgp.device_graph.builder) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"imposter_fused_output_build_graph">>} {
  %s = mogg.shape.create() <shape = #mosh<ape[4]> : !mosh.ape>
  %x = mogg.tensor.create(%px, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %target = mogg.tensor.create(%po, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %out = mogg.tensor.create.epilogue %s (%val : !mogg.simd<f32>, %i : !mogg.dim) capturing {%cap = %target : !mogg.tensor} ptr %po contract = <{role = output, fused = true, access = store, capabilities = none}> device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.tensor.store %cap[%i] = %val : !mogg.tensor, !mogg.dim, !mogg.simd<f32>
  }
  %x1 = mogg.trait.create(%x) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
  mogg.call.build_graph["testing_kernels.testing_kernels.FusedOutputBuildGraph"] inputs(%x1 : !mogg.tensor) outputs(%out : !mogg.tensor) builder(%builder) moOpName = "imposter_fused_output_build_graph" hasEpilogueFusion = true
  mogg.output
}
