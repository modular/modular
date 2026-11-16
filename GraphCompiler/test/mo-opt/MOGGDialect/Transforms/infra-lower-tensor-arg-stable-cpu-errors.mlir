// RUN: not mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mogg-emit-mojo --mogg-dump-mojo -o /dev/null 2>&1 | FileCheck %s
//
// The hand-written `mogg.call.build_graph` names a kernel declaration that
// only exists once `--mogg-import-packages` has run, so parse-time
// verification is disabled; every pass still verifies its output.

// A `Stable` trait argument on a host stub: the stable address slot is device
// memory, so emit refuses to assemble the wrapper.

// CHECK: error: {{.*}}Stable trait argument requires a device kernel; a stable address slot cannot be read on the host

mogg.func @consume_single_build_graph_cpu private dispatch(deferred) (%px: !mogg.pointer, %po: !mogg.pointer, %builder: !mgp.device_graph.builder) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"imposter_stable_single_build_graph">>} {
  %s = mogg.shape.create() <shape = #mosh<ape[4]> : !mosh.ape>
  %x = mogg.tensor.create(%px, %s) <device = "cpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %out = mogg.tensor.create(%po, %s) <device = "cpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
  %x1 = mogg.trait.create(%x) contract = <{role = input, fused = false, access = none, capabilities = none, stable = true}>, device = "cpu" : (!mogg.tensor) -> !mogg.tensor
  %out1 = mogg.trait.create(%out) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "cpu" : (!mogg.tensor) -> !mogg.tensor
  mogg.call.build_graph["testing_kernels.testing_kernels.StableSingleBuildGraph"] inputs(%x1 : !mogg.tensor) outputs(%out1 : !mogg.tensor) builder(%builder) moOpName = "imposter_stable_single_build_graph"
  mogg.output
}
