// RUN: mo-opt --mo-symbolic-optimize %s | FileCheck %s

mo.graph @conv_to_matmul_static_dims(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[1, 1, 3, 9], f32>
) -> !mo.tensor<[32, 10, 10, 9], f32> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 10, 10, 9], f32>

  mo.output %res : !mo.tensor<[32, 10, 10, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_static_dims(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[1, 1, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 10, 10, 9], f32> {
// Reshapes for input/filter into matmul input/weight shape
// CHECK-NEXT:   %[[MATMUL_INPUT:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[32, 10, 10, 3], f32> -> !mo.tensor<[3200, 3], f32>
// CHECK-NEXT:   %[[MATMUL_WEIGHT:.*]] = mo.static.reshape(%[[FILTER]]) : !mo.tensor<[1, 1, 3, 9], f32> -> !mo.tensor<[3, 9], f32>
// Matmul
// CHECK-NEXT:   %[[MATMUL:.*]] = mo.matmul(%[[MATMUL_INPUT]], %[[MATMUL_WEIGHT]])
// CHECK-SAME:     : (!mo.tensor<[3200, 3], f32>, !mo.tensor<[3, 9], f32>) -> !mo.tensor<[3200, 9], f32>
// Reshape for matmul result back to conv output shape
// CHECK-NEXT:   %[[RES:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[3200, 9], f32> -> !mo.tensor<[32, 10, 10, 9], f32>
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 10, 10, 9], f32>


mo.graph @conv_to_matmul_with_device(
  %input: !mo.tensor<[32, 10, 10, 3], f32, dev:0>,
  %filter: !mo.tensor<[1, 1, 3, 9], f32, dev:0>
) -> !mo.tensor<[32, 10, 10, 9], f32, dev:0> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32, dev:0>, !mo.tensor<[1, 1, 3, 9], f32, dev:0>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 10, 10, 9], f32, dev:0>

  mo.output %res : !mo.tensor<[32, 10, 10, 9], f32, dev:0>
}

// CHECK:      mo.graph @conv_to_matmul_with_device(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32, dev:0>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[1, 1, 3, 9], f32, dev:0>
// CHECK-SAME: ) -> !mo.tensor<[32, 10, 10, 9], f32, dev:0> {
// CHECK-NEXT:   %[[MATMUL_INPUT:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[32, 10, 10, 3], f32, dev:0> -> !mo.tensor<[3200, 3], f32, dev:0>
// CHECK-NEXT:   %[[MATMUL_WEIGHT:.*]] = mo.static.reshape(%[[FILTER]]) : !mo.tensor<[1, 1, 3, 9], f32, dev:0> -> !mo.tensor<[3, 9], f32, dev:0>
// CHECK-NEXT:   %[[MATMUL:.*]] = mo.matmul(%[[MATMUL_INPUT]], %[[MATMUL_WEIGHT]])
// CHECK-SAME:     : (!mo.tensor<[3200, 3], f32, dev:0>, !mo.tensor<[3, 9], f32, dev:0>) -> !mo.tensor<[3200, 9], f32, dev:0>
// CHECK-NEXT:   %[[RES:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[3200, 9], f32, dev:0> -> !mo.tensor<[32, 10, 10, 9], f32, dev:0>
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 10, 10, 9], f32, dev:0>

mo.graph @conv_to_matmul_dynamic_dims<N, H, W, CI, CO>(
  %input: !mo.tensor<[N, H, W, CI], f32>,
  %filter: !mo.tensor<[1, 1, CI, CO], f32>
) -> !mo.tensor<[N, H, W, CO], f32> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[N, H, W, CI], f32>, !mo.tensor<[1, 1, CI, CO], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[N, H, W, CO], f32>

  mo.output %res : !mo.tensor<[N, H, W, CO], f32>
}

// CHECK:      mo.graph @conv_to_matmul_dynamic_dims<[[N:.*]], [[H:.*]], [[W:.*]], [[CI:.*]], [[CO:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[N]], [[H]], [[W]], [[CI]]], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[1, 1, [[CI]], [[CO]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32> {
// Reshapes for input/filter into matmul input/weight shape
// CHECK-NEXT:   %[[MATMUL_INPUT:.*]] = mo.static.reshape(%[[INPUT]])
// CHECK-SAME:     : !mo.tensor<[[[N]], [[H]], [[W]], [[CI]]], f32> -> !mo.tensor<[mul_no_wrap([[H]], [[N]], [[W]]), [[CI]]], f32>
// CHECK-NEXT:   %[[MATMUL_WEIGHT:.*]] = mo.static.reshape(%[[FILTER]])
// CHECK-SAME:     : !mo.tensor<[1, 1, [[CI]], [[CO]]], f32> -> !mo.tensor<[[[CI]], [[CO]]], f32>
// Matmul
// CHECK-NEXT:   %[[MATMUL:.*]] = mo.matmul(%[[MATMUL_INPUT]], %[[MATMUL_WEIGHT]]) : (
// CHECK-SAME:     !mo.tensor<[mul_no_wrap([[H]], [[N]], [[W]]), [[CI]]], f32>, !mo.tensor<[[[CI]], [[CO]]], f32>
// CHECK-SAME:   ) -> !mo.tensor<[mul_no_wrap([[H]], [[N]], [[W]]), [[CO]]], f32>
// Reshape for matmul result back to conv output shape
// CHECK-NEXT:   %[[RES:.*]] = mo.static.reshape(%[[MATMUL]])
// CHECK-SAME:     : !mo.tensor<[mul_no_wrap([[H]], [[N]], [[W]]), [[CO]]], f32> -> !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32>
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32>


mo.graph @conv_to_matmul_non_1x1_filter(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[3, 3, 3, 9], f32>
) -> !mo.tensor<[32, 8, 8, 9], f32> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[3, 3, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 8, 8, 9], f32>

  mo.output %res : !mo.tensor<[32, 8, 8, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_non_1x1_filter(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[3, 3, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 8, 8, 9], f32> {
//
// CHECK-NEXT:   %[[ONES:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   %[[PADDINGS:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   %[[NUM_GROUPS:.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
//
// CHECK-NEXT:   %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[ONES]], dilations = %[[ONES]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]], filter_layout = ""] : (
// CHECK-SAME:     !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[3, 3, 3, 9], f32>,
// CHECK-SAME:     !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
// CHECK-SAME:   ) -> !mo.tensor<[32, 8, 8, 9], f32>
//
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 8, 8, 9], f32>
// CHECK-NEXT: }


mo.graph @conv_to_matmul_non_ones_strides(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[1, 1, 3, 9], f32>
) -> !mo.tensor<[32, 10, 5, 9], f32> {

  %strides = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dilations = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 10, 5, 9], f32>

  mo.output %res : !mo.tensor<[32, 10, 5, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_non_ones_strides(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[1, 1, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 10, 5, 9], f32> {
//
// CHECK-NEXT:   %[[STRIDES:.*]] = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   %[[DILATIONS:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   %[[PADDINGS:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   %[[NUM_GROUPS:.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
//
// CHECK-NEXT:   %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]], filter_layout = ""] : (
// CHECK-SAME:     !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
// CHECK-SAME:     !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
// CHECK-SAME:   ) -> !mo.tensor<[32, 10, 5, 9], f32>
//
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 10, 5, 9], f32>
// CHECK-NEXT: }


mo.graph @conv_to_matmul_non_ones_dilations(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[1, 1, 3, 9], f32>
) -> !mo.tensor<[32, 10, 10, 9], f32> {

  %strides = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dilations = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 10, 10, 9], f32>

  mo.output %res : !mo.tensor<[32, 10, 10, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_non_ones_dilations(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[1, 1, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 10, 10, 9], f32> {
// Reshapes for input/filter into matmul input/weight shape
// CHECK-NEXT:   %[[MATMUL_INPUT:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[32, 10, 10, 3], f32> -> !mo.tensor<[3200, 3], f32>
// CHECK-NEXT:   %[[MATMUL_WEIGHT:.*]] = mo.static.reshape(%[[FILTER]]) : !mo.tensor<[1, 1, 3, 9], f32> -> !mo.tensor<[3, 9], f32>
// Matmul
// CHECK-NEXT:   %[[MATMUL:.*]] = mo.matmul(%[[MATMUL_INPUT]], %[[MATMUL_WEIGHT]])
// CHECK-SAME:     : (!mo.tensor<[3200, 3], f32>, !mo.tensor<[3, 9], f32>) -> !mo.tensor<[3200, 9], f32>
// Reshape for matmul result back to conv output shape
// CHECK-NEXT:   %[[RES:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[3200, 9], f32> -> !mo.tensor<[32, 10, 10, 9], f32>
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 10, 10, 9], f32>


mo.graph @conv_to_matmul_non_zero_pads(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[1, 1, 3, 9], f32>
) -> !mo.tensor<[32, 11, 11, 9], f32> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 1, 0, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 11, 11, 9], f32>

  mo.output %res : !mo.tensor<[32, 11, 11, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_non_zero_pads(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[1, 1, 3, 9], f32>
// CHECK-SAME: ) -> !mo.tensor<[32, 11, 11, 9], f32> {
//
// CHECK-NEXT:   %[[ONES:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   %[[PADDINGS:.*]] = mo.constant <#M.dense_array<0, 1, 0, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   %[[NUM_GROUPS:.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
//
// CHECK-NEXT:   %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[ONES]], dilations = %[[ONES]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]], filter_layout = ""] : (
// CHECK-SAME:     !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
// CHECK-SAME:     !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
// CHECK-SAME:   ) -> !mo.tensor<[32, 11, 11, 9], f32>
//
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 11, 11, 9], f32>
// CHECK-NEXT: }

mo.graph @conv_to_matmul_nontrivial_num_groups(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[1, 1, 1, 9], f32>
) -> !mo.tensor<[32, 10, 10, 9], f32> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 1, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 10, 10, 9], f32>

  mo.output %res : !mo.tensor<[32, 10, 10, 9], f32>
}

// CHECK-LABEL: mo.graph @conv_to_matmul_nontrivial_num_groups
// CHECK-NOT: mo.matmul
// CHECK: mo.conv


mo.graph @conv_to_matmul_fcrs_static(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[9, 3, 1, 1], f32, {layout = #mo.layout<FCRS>}>
) -> !mo.tensor<[32, 10, 10, 9], f32> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[9, 3, 1, 1], f32, {layout = #mo.layout<FCRS>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 10, 10, 9], f32>

  mo.output %res : !mo.tensor<[32, 10, 10, 9], f32>
}

// CHECK:      mo.graph @conv_to_matmul_fcrs_static(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[32, 10, 10, 3], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[9, 3, 1, 1], f32, {layout = #mo.layout<FCRS>}>
// CHECK-SAME: ) -> !mo.tensor<[32, 10, 10, 9], f32> {
// Perm constant, then reshapes and transpose for FCRS→KN
// CHECK-NEXT:   %[[PERM:.*]] = mo.constant
// CHECK-NEXT:   %[[MATMUL_INPUT:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[32, 10, 10, 3], f32> -> !mo.tensor<[3200, 3], f32>
// CHECK-NEXT:   %[[RESHAPED:.*]] = mo.static.reshape(%[[FILTER]])
// CHECK-SAME:     -> !mo.tensor<[9, 3], f32
// CHECK-NEXT:   %[[MATMUL_WEIGHT:.*]] = mo.transpose(%[[RESHAPED]], %[[PERM]])
// CHECK-SAME:     -> !mo.tensor<[3, 9], f32
// Matmul — no transpose_b needed
// CHECK-NEXT:   %[[MATMUL:.*]] = mo.matmul(%[[MATMUL_INPUT]], %[[MATMUL_WEIGHT]])
// CHECK-SAME:     -> !mo.tensor<[3200, 9], f32>
// Reshape back to conv output shape
// CHECK-NEXT:   %[[RES:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[3200, 9], f32> -> !mo.tensor<[32, 10, 10, 9], f32>
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[32, 10, 10, 9], f32>


mo.graph @conv_to_matmul_fcrs_dynamic<N, H, W, CI, CO>(
  %input: !mo.tensor<[N, H, W, CI], f32>,
  %filter: !mo.tensor<[CO, CI, 1, 1], f32, {layout = #mo.layout<FCRS>}>
) -> !mo.tensor<[N, H, W, CO], f32> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[N, H, W, CI], f32>, !mo.tensor<[CO, CI, 1, 1], f32, {layout = #mo.layout<FCRS>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[N, H, W, CO], f32>

  mo.output %res : !mo.tensor<[N, H, W, CO], f32>
}

// CHECK:      mo.graph @conv_to_matmul_fcrs_dynamic<[[N:.*]], [[H:.*]], [[W:.*]], [[CI:.*]], [[CO:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[N]], [[H]], [[W]], [[CI]]], f32>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[[[CO]], [[CI]], 1, 1], f32, {layout = #mo.layout<FCRS>}>
// CHECK-SAME: ) -> !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32> {
// CHECK-NEXT:   %[[PERM:.*]] = mo.constant
// CHECK-NEXT:   %[[MATMUL_INPUT:.*]] = mo.static.reshape(%[[INPUT]])
// CHECK-SAME:     -> !mo.tensor<[mul_no_wrap([[H]], [[N]], [[W]]), [[CI]]], f32>
// CHECK-NEXT:   %[[RESHAPED:.*]] = mo.static.reshape(%[[FILTER]])
// CHECK-SAME:     -> !mo.tensor<[[[CO]], [[CI]]], f32
// CHECK-NEXT:   %[[MATMUL_WEIGHT:.*]] = mo.transpose(%[[RESHAPED]], %[[PERM]])
// CHECK-SAME:     -> !mo.tensor<[[[CI]], [[CO]]], f32
// CHECK-NEXT:   %[[MATMUL:.*]] = mo.matmul(%[[MATMUL_INPUT]], %[[MATMUL_WEIGHT]]) : (
// CHECK-SAME:     !mo.tensor<[mul_no_wrap([[H]], [[N]], [[W]]), [[CI]]], f32>, !mo.tensor<[[[CI]], [[CO]]], f32
// CHECK-SAME:   ) -> !mo.tensor<[mul_no_wrap([[H]], [[N]], [[W]]), [[CO]]], f32>
// CHECK-NEXT:   %[[RES:.*]] = mo.static.reshape(%[[MATMUL]])
// CHECK-SAME:     -> !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32>
// CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[[[N]], [[H]], [[W]], [[CO]]], f32>


mo.graph @conv_to_matmul_fcrs_non_1x1_filter(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[9, 3, 3, 3], f32, {layout = #mo.layout<FCRS>}>
) -> !mo.tensor<[32, 8, 8, 9], f32> {

  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %res = mo.conv(%input, %filter) [strides = %ones, dilations = %ones, paddings = %pads, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[9, 3, 3, 3], f32, {layout = #mo.layout<FCRS>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[32, 8, 8, 9], f32>

  mo.output %res : !mo.tensor<[32, 8, 8, 9], f32>
}

// FCRS 3x3 filter should NOT be converted to matmul
// CHECK-LABEL: mo.graph @conv_to_matmul_fcrs_non_1x1_filter
// CHECK-NOT: mo.matmul
// CHECK: mo.conv
