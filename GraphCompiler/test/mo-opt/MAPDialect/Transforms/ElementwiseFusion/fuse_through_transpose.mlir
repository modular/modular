// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// The exp fuses through the transpose into the add. Each load reads an element
// at most once.
// TODO(AGENT-823): the two loads together compute the exp twice per element.

// CHECK-LABEL: mo.graph @fuseThroughTranspose
// CHECK:       map.iter.elementwise<[4, 4]>
// CHECK:       map.load %arg0[%arg4]
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Exp
// CHECK-NEXT:  [[T:%.*]] = map.index.transpose
// CHECK-NEXT:  map.load %arg0[[[T]]]
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Exp
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @fuseThroughTranspose(%x: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32> {
  %e = mo.exp(%x) : !mo.tensor<[4, 4], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%e, %perm) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 4], f32>
  %r = mo.add(%e, %t) : !mo.tensor<[4, 4], f32>
  mo.output %r : !mo.tensor<[4, 4], f32>
}
