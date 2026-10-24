// RUN: mo-opt %s --mo-infer-layouts | FileCheck %s


// CHECK-LABEL: mo.graph @chained_conv2d_with_reused_filter(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FCRS>}>,
// CHECK-SAME:   %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[4], si64>,
// CHECK-SAME:   %[[NUM_GROUPS:.*]]: !mo.tensor<[], si64>
// CHECK-SAME: ) -> (!mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>, !mo.tensor<[?, ?, ?, ?], f32>) {
mo.graph @chained_conv2d_with_reused_filter(
  %input: !mo.tensor<[?, ?, ?, ?], f32>,
  %filter: !mo.tensor<[?, ?, ?, ?], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>,
  %num_groups: !mo.tensor<[], si64>
) -> (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) {

  // CHECK-NEXT: %[[CONV1:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]], filter_layout = ""] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FCRS>}>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[4], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
  %conv1 = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: %[[CONV2:.*]] = mo.conv(%[[CONV1]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]], filter_layout = ""] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FCRS>}>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[4], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?], f32>
  %conv2 = mo.conv(%conv1, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[CONV1]], %[[CONV2]] : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>, !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %conv1, %conv2 : !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK-LABEL: mo.graph @chained_conv3d_with_reused_filter(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<FCQRS>}>,
// CHECK-SAME:   %[[STRIDES:.*]]: !mo.tensor<[3], si64>, %[[DILATIONS:.*]]: !mo.tensor<[3], si64>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[6], si64>,
// CHECK-SAME:   %[[NUM_GROUPS:.*]]: !mo.tensor<[], si64>
// CHECK-SAME: ) -> (!mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>, !mo.tensor<[?, ?, ?, ?, ?], f32>) {
mo.graph @chained_conv3d_with_reused_filter(
  %input: !mo.tensor<[?, ?, ?, ?, ?], f32>,
  %filter: !mo.tensor<[?, ?, ?, ?, ?], f32>,
  %strides: !mo.tensor<[3], si64>,
  %dilations: !mo.tensor<[3], si64>,
  %paddings: !mo.tensor<[6], si64>,
  %num_groups: !mo.tensor<[], si64>
) -> (!mo.tensor<[?, ?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?, ?], f32>) {

  // CHECK-NEXT: %[[CONV1:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]], filter_layout = ""] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<FCQRS>}>,
  // CHECK-SAME:   !mo.tensor<[3], si64>,
  // CHECK-SAME:   !mo.tensor<[3], si64>,
  // CHECK-SAME:   !mo.tensor<[6], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>
  %conv1 = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?, ?], f32>,
    !mo.tensor<[3], si64>,
    !mo.tensor<[3], si64>,
    !mo.tensor<[6], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?, ?], f32>

  // CHECK-NEXT: %[[CONV2:.*]] = mo.conv(%[[CONV1]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]], filter_layout = ""] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<FCQRS>}>,
  // CHECK-SAME:   !mo.tensor<[3], si64>,
  // CHECK-SAME:   !mo.tensor<[3], si64>,
  // CHECK-SAME:   !mo.tensor<[6], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?, ?], f32>
  %conv2 = mo.conv(%conv1, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?, ?], f32>,
    !mo.tensor<[3], si64>,
    !mo.tensor<[3], si64>,
    !mo.tensor<[6], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[CONV1]], %[[CONV2]] : !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>, !mo.tensor<[?, ?, ?, ?, ?], f32>
  mo.output %conv1, %conv2 : !mo.tensor<[?, ?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?, ?], f32>
}

// CHECK-LABEL: mo.graph @chained_conv2d_transpose_with_reused_filter(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<RSFC>}>,
// CHECK-SAME:   %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[4], si64>,
// CHECK-SAME:   %[[OUTPUT_PADDINGS:.*]]: !mo.tensor<[2], si64>
// CHECK-SAME: ) -> (!mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>, !mo.tensor<[?, ?, ?, ?], f32>) {
mo.graph @chained_conv2d_transpose_with_reused_filter(
  %input: !mo.tensor<[?, ?, ?, ?], f32>,
  %filter: !mo.tensor<[?, ?, ?, ?], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>,
  %output_paddings: !mo.tensor<[2], si64>
) -> (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) {

  // CHECK-NEXT: %[[CONV_TRANSPOSE1:.*]] = mo.conv_transpose(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], output_paddings = %[[OUTPUT_PADDINGS]], filter_layout = ""] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<RSFC>}>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[4], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
  %conv_transpose1 = mo.conv_transpose(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, output_paddings = %output_paddings, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[2], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: %[[CONV_TRANSPOSE2:.*]] = mo.conv_transpose(%[[CONV_TRANSPOSE1]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], output_paddings = %[[OUTPUT_PADDINGS]], filter_layout = ""] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<RSFC>}>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[4], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?], f32>
  %conv_transpose2 = mo.conv_transpose(%conv_transpose1, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, output_paddings = %output_paddings, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[2], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[CONV_TRANSPOSE1]], %[[CONV_TRANSPOSE2]] : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>, !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %conv_transpose1, %conv_transpose2 : !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK-LABEL: mo.graph @chained_conv3d_transpose_with_reused_filter(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<QRSFC>}>,
// CHECK-SAME:   %[[STRIDES:.*]]: !mo.tensor<[3], si64>, %[[DILATIONS:.*]]: !mo.tensor<[3], si64>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[6], si64>,
// CHECK-SAME:   %[[OUTPUT_PADDINGS:.*]]: !mo.tensor<[3], si64>
// CHECK-SAME: ) -> (!mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>, !mo.tensor<[?, ?, ?, ?, ?], f32>) {
mo.graph @chained_conv3d_transpose_with_reused_filter(
  %input: !mo.tensor<[?, ?, ?, ?, ?], f32>,
  %filter: !mo.tensor<[?, ?, ?, ?, ?], f32>,
  %strides: !mo.tensor<[3], si64>,
  %dilations: !mo.tensor<[3], si64>,
  %paddings: !mo.tensor<[6], si64>,
  %output_paddings: !mo.tensor<[3], si64>
) -> (!mo.tensor<[?, ?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?, ?], f32>) {

  // CHECK-NEXT: %[[CONV_TRANSPOSE1:.*]] = mo.conv_transpose(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], output_paddings = %[[OUTPUT_PADDINGS]], filter_layout = ""] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<QRSFC>}>,
  // CHECK-SAME:   !mo.tensor<[3], si64>,
  // CHECK-SAME:   !mo.tensor<[3], si64>,
  // CHECK-SAME:   !mo.tensor<[6], si64>,
  // CHECK-SAME:   !mo.tensor<[3], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>
  %conv_transpose1 = mo.conv_transpose(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, output_paddings = %output_paddings, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?, ?], f32>,
    !mo.tensor<[3], si64>,
    !mo.tensor<[3], si64>,
    !mo.tensor<[6], si64>,
    !mo.tensor<[3], si64>
  ) -> !mo.tensor<[?, ?, ?, ?, ?], f32>

  // CHECK-NEXT: %[[CONV_TRANSPOSE2:.*]] = mo.conv_transpose(%[[CONV_TRANSPOSE1]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], output_paddings = %[[OUTPUT_PADDINGS]], filter_layout = ""] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<QRSFC>}>,
  // CHECK-SAME:   !mo.tensor<[3], si64>,
  // CHECK-SAME:   !mo.tensor<[3], si64>,
  // CHECK-SAME:   !mo.tensor<[6], si64>,
  // CHECK-SAME:   !mo.tensor<[3], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?, ?], f32>
  %conv_transpose2 = mo.conv_transpose(%conv_transpose1, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, output_paddings = %output_paddings, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?, ?], f32>,
    !mo.tensor<[3], si64>,
    !mo.tensor<[3], si64>,
    !mo.tensor<[6], si64>,
    !mo.tensor<[3], si64>
  ) -> !mo.tensor<[?, ?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[CONV_TRANSPOSE1]], %[[CONV_TRANSPOSE2]] : !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<NDHWC>}>, !mo.tensor<[?, ?, ?, ?, ?], f32>
  mo.output %conv_transpose1, %conv_transpose2 : !mo.tensor<[?, ?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?, ?], f32>
}

// CHECK-LABEL: mo.graph @matmul_dyn_weights_in_output(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?], f32>,
// CHECK-SAME:   %[[W1:.*]]: !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>,
// CHECK-SAME:   %[[W2:.*]]: !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>
// CHECK-SAME: ) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>) {
mo.graph @matmul_dyn_weights_in_output(
  %input: !mo.tensor<[?, ?], f32>,
  %w1: !mo.tensor<[?, ?], f32>,
  %w2: !mo.tensor<[?, ?], f32>
) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) {

  // CHECK-NEXT: %[[WEIGHTS:.*]] = mo.add(%[[W1]], %[[W2]]) : !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>
  %weights = mo.add(%w1, %w2) : !mo.tensor<[?, ?], f32>

  // CHECK-NEXT: %[[MATMUL:.*]] = mo.matmul(%[[INPUT]], %[[WEIGHTS]]) : (
  // CHECK-SAME:   !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?], f32>
  %matmul = mo.matmul(%input, %weights) : (
    !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>
  ) -> !mo.tensor<[?, ?], f32>

  // CHECK-NEXT: mo.output %[[MATMUL]], %[[WEIGHTS]] : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>
  mo.output %matmul, %weights : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>
}
