// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-elementwise --map-outline-structured-kernel-bodies --map-to-mogg --mo-nan-check-pass="stride=1" --mogg-outline-routines -mogg-add-try-except-blocks | FileCheck %s

// A MAP ABI stub gets mogg.try/mogg.raise like a MOGG one. Without it there
// is no mogg.raise on this path, and its emitMojo is the only renderer of a
// kernel error's name, fusion info and source traceback.

// CHECK-LABEL: mogg.func @__add private
// CHECK:         mogg.try {
// CHECK:         } except(%{{[^)]+}}) {
// CHECK:           mogg.raise

mo.graph @checked() -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[4], f32>
  %1 = mo.add(%0, %0) : !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}
