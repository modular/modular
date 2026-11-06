// RUN: rm -rf %t && mkdir -p %t
// RUN: not env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// A kernel that declares the retired `target` parameter is rejected, with a
// message pointing at `ctx.target`.
// CHECK: the `target` kernel parameter is no longer supported; take a `ctx: DeviceContext` argument and use `ctx.target` instead
mo.graph @target_param(%a: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %c = mo.custom [symbol = "kernel_with_target_param", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  mo.output %c : !mo.tensor<[2], f32>
}
