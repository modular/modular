// RUN: mo-opt %s --canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// LayoutTransformOp
//===----------------------------------------------------------------------===//

// CHECK:      mo.graph @layout_transform_different_layout(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
// CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}> {
mo.graph @layout_transform_different_layout(
  %input: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
) -> !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}> {

  // CHECK-NEXT: %[[RES:.*]] = mo.layout.transform(%[[INPUT]]) :
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}> ->
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>
  %res = mo.layout.transform(%input) :
    !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}> ->
    !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>

  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>
  mo.output %res : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>
}

// CHECK:      mo.graph @layout_transform_same_layout_folded(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
// CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}> {
mo.graph @layout_transform_same_layout_folded(
  %input: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
) -> !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}> {

  %res = mo.layout.transform(%input) :
    !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}> ->
    !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>

  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
  mo.output %res : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
}
