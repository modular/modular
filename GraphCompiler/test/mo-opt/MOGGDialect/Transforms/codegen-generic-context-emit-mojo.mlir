// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo

// A kernel taking `ctx: DeviceContext` and no `target` parameter is handed the
// stub's context as is: the stub's own context is generic over its target, so
// the caller picks it (see `unpack_device_ctx`).
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[PC:.*]]: {{.*}}DeviceContext) raises:
// CHECK-NOT: assert_cpu_target
// CHECK:   {{.*}}.MapLowerGenericContext.execute({{.*}}, {{.*}}, [[PC]])
mo.graph @cpu(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_generic_context", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// A bare `ctx: DeviceContext` is generic over its target as well.
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[PC2:.*]]: {{.*}}DeviceContext) raises:
// CHECK-NOT: assert_cpu_target
// CHECK:   {{.*}}.MapLowerOnePlainOutput.execute({{.*}}, {{.*}}, [[PC2]])
mo.graph @bare(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
