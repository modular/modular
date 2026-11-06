// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo

// A declarative kernel can update a buffer in place, as a KV cache store does.
// The mutable input reaches the launch as a `!mogg.buffer` and crosses it like
// any tensor: as its `TileTensor` projection, rebuilt on the device.

// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}, [[CTX:[A-Za-z0-9_]+]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[CONFIG:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarativeInplace.execute_configuration([[OUT:[A-Za-z0-9_]+]], [[IN:[A-Za-z0-9_]+]], [[CACHE:[A-Za-z0-9_]+]])
// CHECK:   comptime [[OUT_T:[A-Za-z0-9_]+]] = type_of([[OUT]])
// CHECK:   comptime [[IN_T:[A-Za-z0-9_]+]] = type_of([[IN]])
// CHECK:   comptime [[CACHE_T:[A-Za-z0-9_]+]] = type_of([[CACHE]])
// CHECK:   def [[KERNEL:execute_device_[_0-9]+]]([[A0:[A-Za-z0-9_]+]]: [[OUT_T]].TileTensorType, [[A1:[A-Za-z0-9_]+]]: [[IN_T]].TileTensorType, [[A2:[A-Za-z0-9_]+]]: [[CACHE_T]].TileTensorType):
// CHECK-NEXT: {{.*}}.MapLowerDeclarativeInplace.execute_device([[OUT_T]]([[A0]]), [[IN_T]]([[A1]]), [[CACHE_T]]([[A2]]))
// CHECK:   builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}([[CTX]], [[CONFIG]], [[OUT]].to_tile_tensor(), [[IN]].to_tile_tensor(), [[CACHE]].to_tile_tensor())

mo.graph @declarative_mutable_input(%a: !mo.tensor<[2, 2], f32>, %cache: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.chain) {
  %chain = mo.chain.create()
  %c, %out_chain = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%a, %cache, %chain) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  mo.output %c, %out_chain : !mo.tensor<[2, 2], f32>, !mo.chain
}
