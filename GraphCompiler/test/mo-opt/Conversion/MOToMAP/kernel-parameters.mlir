// RUN: mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Kernel parameters that do not arrive as plain typed attributes on the MO op.
// The legacy MO->MOGG lowering supplies both of these, so a MAP call that
// omits them fails later with "Failed to infer parameter".

// Conv's strides, dilations and paddings are SSA operands, not attributes, so
// they are only reachable by constant-folding the operands.

// CHECK-LABEL: mo.graph @convStaticStrides
// CHECK: map.call.opaque @{{.*}}ConvCPU[
// CHECK-SAME: static_dilations = [1, 1]
// CHECK-SAME: static_padding = [0, 0, 0, 0]
// CHECK-SAME: static_strides = [2, 2]
mo.graph @convStaticStrides(%input: !mo.tensor<[1, 5, 5, 3], f32>, %filter: !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>) -> !mo.tensor<[1, 2, 2, 1], f32> {
  %st = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %res = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 3], f32>, !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 2, 2, 1], f32>
  mo.output %res : !mo.tensor<[1, 2, 2, 1], f32>
}

// A non-constant stride operand cannot be folded, so it becomes the dynamic
// sentinel rather than being dropped.

// CHECK-LABEL: mo.graph @convDynamicStrides
// CHECK: map.call.opaque @{{.*}}ConvCPU[
// CHECK-SAME: static_strides = [-9223372036854775808, -9223372036854775808]
mo.graph @convDynamicStrides(%input: !mo.tensor<[1, 5, 5, 3], f32>, %filter: !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>, %st: !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2, 2, 1], f32> {
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %res = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 3], f32>, !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 2, 2, 1], f32>
  mo.output %res : !mo.tensor<[1, 2, 2, 1], f32>
}

// `coordinate_transform_mode` is an enum attribute, so it is not a TypedAttr
// and needs converting to the integer the kernel declares.

// CHECK-LABEL: mo.graph @resizeCoordTransformMode
// CHECK: map.call.opaque @{{.*}}ResizeNearest[
// CHECK-SAME: coordinate_transform_mode = 0 : index
mo.graph @resizeCoordTransformMode(%input: !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
  %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %res = mo.resize.nearest(%input, %size) [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel>] : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
  mo.output %res : !mo.tensor<[1, 1, 7, 8], f32>
}
