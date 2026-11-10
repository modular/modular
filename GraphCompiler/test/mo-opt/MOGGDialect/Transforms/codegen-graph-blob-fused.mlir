// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-form-graph-blobs --mogg-outline-routines --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo

// Two declarative calls with fused outputs form one graph blob. Each member's
// fused output crosses the launch as its `device_type`, with the bias its
// epilogue captures inside it, and the kernel uses it as is. The first
// member's result is the second member's input, which the kernel rebuilds from
// its `TileTensor`.

// CHECK: var [[OUT0:[A-Za-z0-9_]+]] = _MoggOutputStoreFusionTensor[{{.*}}, True, False, False]
// CHECK: comptime [[OUT0_T:[A-Za-z0-9_]+]] = type_of([[OUT0]])
// CHECK: comptime [[IN0_T:[A-Za-z0-9_]+]] = type_of([[IN0:[A-Za-z0-9_]+]])
// CHECK: var [[OUT1:[A-Za-z0-9_]+]] = _MoggOutputStoreFusionTensor[{{.*}}, True, False, False]
// CHECK: comptime [[OUT1_T:[A-Za-z0-9_]+]] = type_of([[OUT1]])
// CHECK: comptime [[IN1_T:[A-Za-z0-9_]+]] = type_of([[IN1:[A-Za-z0-9_]+]])
// CHECK: def [[KERNEL:execute_device_[_0-9]+]]([[A0:[A-Za-z0-9_]+]]: [[OUT0_T]].device_type, [[A1:[A-Za-z0-9_]+]]: [[IN0_T]].device_type, [[A2:[A-Za-z0-9_]+]]: [[OUT1_T]].device_type, [[A3:[A-Za-z0-9_]+]]: [[IN1_T]].device_type, {{.*}}):
// CHECK:   {{.*}}.MapLowerDeclarativeFusedStore.execute_device([[A0]], [[IN0_T]]([[A1]]))
// CHECK:   .barrier()
// CHECK:   {{.*}}.MapLowerDeclarativeFusedStore.execute_device([[A2]], [[IN1_T]]([[A3]]))
// CHECK: builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}({{.*}}, [[OUT0]], [[IN0]], [[OUT1]], [[IN1]], {{.*}})

mo.graph @blob(%a: !mo.tensor<[2, 2], f32, gpu:0>, %bias: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %c = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %d = mo.add(%c, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %e = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%d) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %f = mo.add(%e, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %f : !mo.tensor<[2, 2], f32, gpu:0>
}
