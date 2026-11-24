// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo

// An elementwise op fuses into a declarative kernel's operand. The fused
// operand holds a fusion functor, not one buffer, so its `device_type` is the
// access-enabled view on which the fusion runs rather than a `TileTensor`
// projection, and the kernel uses it as is instead of rebuilding it. The same
// host value feeds `execute_configuration`.

// A bias add fuses into the output's store.
// CHECK-LABEL: def __custom__map_lower_declarative_fused_store_{{[0-9a-f]+}}(
// CHECK:   var [[OUT:[A-Za-z0-9_]+]] = _MoggOutputStoreFusionTensor[DType.float32, {{.*}}, True, False, False]
// CHECK:   var [[CONFIG:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarativeFusedStore.execute_configuration([[OUT]], [[IN:[A-Za-z0-9_]+]])
// CHECK-NEXT: comptime [[OUT_T:[A-Za-z0-9_]+]] = type_of([[OUT]])
// CHECK-NEXT: comptime [[IN_T:[A-Za-z0-9_]+]] = type_of([[IN]])
// CHECK-NEXT: def [[KERNEL:execute_device_[_0-9]+]]([[A0:[A-Za-z0-9_]+]]: [[OUT_T]].device_type, [[A1:[A-Za-z0-9_]+]]: [[IN_T]].device_type):
// CHECK-NEXT:   {{.*}}.MapLowerDeclarativeFusedStore.execute_device([[A0]], [[IN_T]]([[A1]]))
// CHECK:   builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}({{.*}}, [[CONFIG]], [[OUT]], [[IN]])

// A multiply fuses into the input's load.
// CHECK-LABEL: def __mul_{{[0-9a-f]+}}(
// CHECK:   var [[IN:[A-Za-z0-9_]+]] = _MoggInputLoadFusionTensor[DType.float32, {{.*}}, True, False]
// CHECK:   var [[CONFIG:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarativeFusedLoad.execute_configuration([[OUT:[A-Za-z0-9_]+]], [[IN]])
// CHECK-NEXT: comptime [[OUT_T:[A-Za-z0-9_]+]] = type_of([[OUT]])
// CHECK-NEXT: comptime [[IN_T:[A-Za-z0-9_]+]] = type_of([[IN]])
// CHECK-NEXT: def [[KERNEL:execute_device_[_0-9]+]]([[A0:[A-Za-z0-9_]+]]: [[OUT_T]].device_type, [[A1:[A-Za-z0-9_]+]]: [[IN_T]].device_type):
// CHECK-NEXT:   {{.*}}.MapLowerDeclarativeFusedLoad.execute_device([[OUT_T]]([[A0]]), [[A1]])
// CHECK:   builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}({{.*}}, [[CONFIG]], [[OUT]], [[IN]])

// A dynamic dim of the fused output's shape travels inside the fused operand's
// layout, so it isn't a launch argument of its own.
// CHECK-LABEL: def __custom__map_lower_declarative_fused_store_{{[0-9a-f]+}}_0(
// CHECK:   var [[OUT:[A-Za-z0-9_]+]] = _MoggOutputStoreFusionTensor[
// CHECK:   def [[KERNEL:execute_device_[_0-9]+]]({{[A-Za-z0-9_]+}}: {{[A-Za-z0-9_]+}}.device_type, {{[A-Za-z0-9_]+}}: {{[A-Za-z0-9_]+}}.device_type):
// CHECK:   builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}({{.*}}, [[OUT]], {{[A-Za-z0-9_]+}})

mo.graph @fused_store(%a: !mo.tensor<[2, 2], f32, gpu:0>, %bias: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %c = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %d = mo.add(%c, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %d : !mo.tensor<[2, 2], f32, gpu:0>
}

mo.graph @fused_load(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %x = mo.mul(%a, %b) : !mo.tensor<[2, 2], f32, gpu:0>
  %c = mo.custom [symbol = "map_lower_declarative_fused_load", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %c : !mo.tensor<[2, 2], f32, gpu:0>
}

mo.graph @fused_store_dynamic(%a: !mo.tensor<[batch, 2], f32, gpu:0>, %bias: !mo.tensor<[batch, 2], f32, gpu:0>) -> !mo.tensor<[batch, 2], f32, gpu:0> {
  %c = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[batch, 2], f32, gpu:0>) -> !mo.tensor<[batch, 2], f32, gpu:0>
  %d = mo.add(%c, %bias) : !mo.tensor<[batch, 2], f32, gpu:0>
  mo.output %d : !mo.tensor<[batch, 2], f32, gpu:0>
}
