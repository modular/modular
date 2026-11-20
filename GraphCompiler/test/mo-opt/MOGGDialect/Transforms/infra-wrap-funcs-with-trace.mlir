// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-elementwise --map-outline-structured-kernel-bodies --map-to-mogg --mogg-outline-routines --mogg-wrap-funcs-with-trace | FileCheck %s

// A MAP ABI stub's body gets wrapped in mogg.with_trace. The terminator stays
// outside and reads its results, so a value-returning stub keeps its return.

// CHECK-LABEL: mogg.func @__add
// CHECK:         mogg.with_trace {
// CHECK:           mogg.call.foreach
// CHECK:           mogg.yield
// CHECK:         }
// CHECK:         mogg.output

mo.graph @traced() -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[4], f32>
  %1 = mo.add(%0, %0) : !mo.tensor<[4], f32>
  mo.output %1 : !mo.tensor<[4], f32>
}
