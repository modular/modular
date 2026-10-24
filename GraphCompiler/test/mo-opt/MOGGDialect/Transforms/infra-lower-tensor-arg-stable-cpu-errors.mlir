// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-pipeline --map-to-mogg --mogg-pipeline --mogg-to-mgp --mogg-outline-async-routines --mogg-add-try-except-blocks --mogg-emit-mojo --mogg-dump-mojo -o /dev/null 2>&1 | FileCheck %s

// A `Stable` trait argument on a host kernel: the stable address slot is device
// memory, so emit refuses to assemble the wrapper.

// CHECK: error: {{.*}}Stable trait argument requires a device kernel; a stable address slot cannot be read on the host

mo.graph @trait_stable_negate_cpu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %z = mo.custom [symbol = "trait_stable_negate", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %z : !mo.tensor<[4], f32>
}
