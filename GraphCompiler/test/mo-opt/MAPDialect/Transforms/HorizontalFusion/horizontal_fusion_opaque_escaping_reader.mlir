// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-horizontal | FileCheck %s

// An opaque kernel's result read by an elementwise consumer and by the graph
// output stays a plain producer: the consumer is not merged into its epilogue.

// CHECK-LABEL: mo.graph @matmul_result_escapes
// CHECK:       %[[MM:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs
// CHECK-NEXT:  outputs (%{{[a-z0-9]+}} : !mo.tensor<[8, 8], f32>) {
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK:       %[[ADD:.*]] = map.iter.elementwise
// CHECK-NEXT:  inputs (%[[MM]]: !mo.tensor<[8, 8], f32>, %{{.*}}: !mo.tensor<[8, 8], f32>)
// CHECK:       mo.output %[[MM]], %[[ADD]]
mo.graph @matmul_result_escapes(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  mo.output %mm, %a : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}

// Two elementwise consumers of the same opaque result still merge with each
// other, but not into the producer.

// CHECK-LABEL: mo.graph @matmul_result_escapes_two_consumers
// CHECK:       %[[MM2:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs
// CHECK-NEXT:  outputs (%{{[a-z0-9]+}} : !mo.tensor<[8, 8], f32>) {
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK:       %[[ADDS:.*]]:2 = map.iter.elementwise
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[MM2]], %[[ADDS]]#0, %[[ADDS]]#1
mo.graph @matmul_result_escapes_two_consumers(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[8, 8], f32>
  mo.output %mm, %a, %b : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}
