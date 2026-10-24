// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo

// A kernel with `execute_device` and `execute_configuration` is launched by
// the graph compiler: `execute_configuration`, whose defaulted parameter is
// left to its default, gives the grid and block, and a kernel taking each
// tensor as its `TileTensor` projection rebuilds the operands and calls
// `execute_device`.

// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}, [[CTX:[A-Za-z0-9_]+]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[CONFIG:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration([[OUT:[A-Za-z0-9_]+]], [[IN:[A-Za-z0-9_]+]])
// CHECK:   comptime [[OUT_T:[A-Za-z0-9_]+]] = type_of([[OUT]])
// CHECK:   comptime [[IN_T:[A-Za-z0-9_]+]] = type_of([[IN]])
// CHECK:   def [[KERNEL:execute_device_[_0-9]+]]([[A0:[A-Za-z0-9_]+]]: [[OUT_T]].TileTensorType, [[A1:[A-Za-z0-9_]+]]: [[IN_T]].TileTensorType):
// CHECK-NEXT: {{.*}}.MapLowerDeclarative.execute_device([[OUT_T]]([[A0]]), [[IN_T]]([[A1]]))
// CHECK:   var [[FUNC:[A-Za-z0-9_]+]] = [[CTX]].compile_function{{\[}}[[KERNEL]]{{\]}}()
// CHECK:   [[CTX]].enqueue_function([[FUNC]], [[OUT]].to_tile_tensor(), [[IN]].to_tile_tensor(), grid_dim = [[CONFIG]].get_grid_dim(), block_dim = [[CONFIG]].get_block_dim())

mo.graph @declarative(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
