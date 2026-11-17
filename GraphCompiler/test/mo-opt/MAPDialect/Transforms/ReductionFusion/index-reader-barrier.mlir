// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-reductions | FileCheck %s

// An index reader does not fuse into a fold that reads it through a transpose.
// Fused, it would see the fold's index.

// CHECK-LABEL: mo.graph @indexReaderThroughTranspose
// CHECK:       map.iter.elementwise<[128, 8]>
// CHECK:       map.call.elementwise @{{.*}}.ElemwiseAddWithIndex
// CHECK:       map.iter.rowwise<[8, 128], axis = 1>
// CHECK:       map.index.transpose
// CHECK-NOT:   ElemwiseAddWithIndex
// CHECK:       mo.output
mo.graph @indexReaderThroughTranspose(%x: !mo.tensor<[128, 8], si32>) -> !mo.tensor<[8, 1], si32> {
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[128, 8], si32>) -> !mo.tensor<[128, 8], si32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%z, %perm) : (!mo.tensor<[128, 8], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 128], si32>
  %r = mo.reduce.add(%t) : (!mo.tensor<[8, 128], si32>) -> !mo.tensor<[8, 1], si32>
  mo.output %r : !mo.tensor<[8, 1], si32>
}
