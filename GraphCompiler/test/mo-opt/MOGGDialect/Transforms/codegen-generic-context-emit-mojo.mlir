// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo

// A kernel taking `ctx: DeviceContext.Generic` and no `target` parameter is
// handed a CPU-typed context when it runs on the CPU, instead of the default
// accelerator's.
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[PC:.*]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[CPU_CTX:.*]] = [[PC]].assert_cpu_target()
// CHECK:   {{.*}}.MapLowerGenericContext.execute({{.*}}, {{.*}}, [[CPU_CTX]])
mo.graph @cpu(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_generic_context", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// A kernel with a concrete `ctx: DeviceContext` keeps receiving the stub's
// context untouched.
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[PC2:.*]]: {{.*}}DeviceContext) raises:
// CHECK-NOT: assert_cpu_target
// CHECK:   {{.*}}.MapLowerOnePlainOutput.execute[target = "cpu"]({{.*}}, {{.*}}, [[PC2]])
mo.graph @concrete(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
