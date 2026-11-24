// RUN: mo-opt --mo-symbolic-optimize %s | FileCheck %s

// CHECK-LABEL: mo.graph
// CHECK-SAME: @success_const_rhs([[VARG0:[^ ]*]]: !mo.tensor<[16, 64, 128, 128], f32>) -> !mo.tensor<[1, 1024, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.sub([[VARG0]], [[V1]]) : !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.reshape([[V2]]) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[1, 1024, 128, 128], f32>
mo.graph @success_const_rhs(
  %sri: !mo.tensor<[16, 64, 128, 128], f32>
) -> !mo.tensor<[1, 1024, 128, 128], f32> {
  %pc = mo.constant <#M.dense_array<1.0> : !M.array<1xf32>> : !mo.tensor<[], f32>

  %b0 = mo.static.broadcast_to(%pc) : !mo.tensor<[], f32> -> !mo.tensor<[1024], f32>
  %b1 = mo.static.reshape(%b0) : !mo.tensor<[1024], f32> -> !mo.tensor<[1, 1024, 1, 1], f32>

  %lhs = mo.static.reshape(%sri) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %rhs = mo.static.broadcast_to(%b1) : !mo.tensor<[1, 1024, 1, 1], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %out = mo.sub(%lhs, %rhs) : !mo.tensor<[1, 1024, 128, 128], f32>

  mo.output %out : !mo.tensor<[1, 1024, 128, 128], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @success_const_rhs_external([[VARG0:[^ ]*]]: !mo.tensor<[16, 64, 128, 128], f32>) -> !mo.tensor<[1, 1024, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant.external [name = "blob", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.sub([[VARG0]], [[V1]]) : !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.reshape([[V2]]) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[1, 1024, 128, 128], f32>
mo.graph @success_const_rhs_external(
  %sri: !mo.tensor<[16, 64, 128, 128], f32>
) -> !mo.tensor<[1, 1024, 128, 128], f32> {
  %pc = mo.constant.external [name = "blob", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[], f32>

  %b0 = mo.static.broadcast_to(%pc) : !mo.tensor<[], f32> -> !mo.tensor<[1024], f32>
  %b1 = mo.static.reshape(%b0) : !mo.tensor<[1024], f32> -> !mo.tensor<[1, 1024, 1, 1], f32>

  %lhs = mo.static.reshape(%sri) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %rhs = mo.static.broadcast_to(%b1) : !mo.tensor<[1, 1024, 1, 1], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %out = mo.sub(%lhs, %rhs) : !mo.tensor<[1, 1024, 128, 128], f32>

  mo.output %out : !mo.tensor<[1, 1024, 128, 128], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @success_const_lhs([[VARG0:[^ ]*]]: !mo.tensor<[16, 64, 128, 128], f32>) -> !mo.tensor<[1, 1024, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.sub([[V1]], [[VARG0]]) : !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.reshape([[V2]]) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[1, 1024, 128, 128], f32>
mo.graph @success_const_lhs(
  %sri: !mo.tensor<[16, 64, 128, 128], f32>
) -> !mo.tensor<[1, 1024, 128, 128], f32> {
  %pc = mo.constant <#M.dense_array<1.0> : !M.array<1xf32>> : !mo.tensor<[], f32>

  %b0 = mo.static.broadcast_to(%pc) : !mo.tensor<[], f32> -> !mo.tensor<[1024], f32>
  %b1 = mo.static.reshape(%b0) : !mo.tensor<[1024], f32> -> !mo.tensor<[1, 1024, 1, 1], f32>

  %lhs = mo.static.broadcast_to(%b1) : !mo.tensor<[1, 1024, 1, 1], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %rhs = mo.static.reshape(%sri) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %out = mo.sub(%lhs, %rhs) : !mo.tensor<[1, 1024, 128, 128], f32>

  mo.output %out : !mo.tensor<[1, 1024, 128, 128], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @failure_non_const_subtree([[VARG0:[^ ]*]]: !mo.tensor<[16, 128, 128, 8], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[7, 7, 8, 64], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[1, 1024, 1, 1], f32>, [[VARG3:[^ ]*]]: !mo.tensor<[1, 1024, 1, 1], f32>) -> !mo.tensor<[1, 1024, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<1, 1> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant <#M.dense_array<3, 3, 3, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.conv([[VARG0]], [[VARG1]]) [strides = [[V1]], dilations = [[V1]], paddings = [[V2]], num_groups = [[V0]], filter_layout = ""] : (!mo.tensor<[16, 128, 128, 8], f32>, !mo.tensor<[7, 7, 8, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[16, 128, 128, 64], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.transpose([[V4]], [[V3]]) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.static.reshape([[V5]]) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
// CHECK-NEXT: [[V7:[^ ]*]] = mo.static.broadcast_to([[VARG3]]) : !mo.tensor<[1, 1024, 1, 1], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
// CHECK-NEXT: [[V8:[^ ]*]] = mo.sub([[V6]], [[V7]]) : !mo.tensor<[1, 1024, 128, 128], f32>
// CHECK-NEXT: mo.output [[V8]] : !mo.tensor<[1, 1024, 128, 128], f32>
mo.graph @failure_non_const_subtree(
  %35: !mo.tensor<[16, 128, 128, 8], f32>,
  %23: !mo.tensor<[7, 7, 8, 64], f32>,
  %21: !mo.tensor<[1, 1024, 1, 1], f32>,
  %22: !mo.tensor<[1, 1024, 1, 1], f32>
) -> !mo.tensor<[1, 1024, 128, 128], f32> {
  %25 = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  %26 = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
  %27 = mo.constant <#M.dense_array<1, 1> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %28 = mo.constant <#M.dense_array<3, 3, 3, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
  %29 = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>

  %36 = mo.conv(%35, %23) [strides = %27, dilations = %27, paddings = %28, num_groups = %25, filter_layout = ""] : (!mo.tensor<[16, 128, 128, 8], f32>, !mo.tensor<[7, 7, 8, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[16, 128, 128, 64], f32>
  %37 = mo.transpose(%36, %29) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>

  %38 = mo.static.reshape(%37) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %39 = mo.static.broadcast_to(%22) : !mo.tensor<[1, 1024, 1, 1], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %40 = mo.sub(%38, %39) : !mo.tensor<[1, 1024, 128, 128], f32>

  mo.output %40 : !mo.tensor<[1, 1024, 128, 128], f32>
}


// CHECK-LABEL: mo.graph
// CHECK-SAME: @success_reshape_both_sides([[VARG0:[^ ]*]]: !mo.tensor<[16, 64, 128, 128], f32>) -> !mo.tensor<[1, 1024, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.sub([[VARG0]], [[V1]]) : !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.reshape([[V2]]) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[1, 1024, 128, 128], f32>
mo.graph @success_reshape_both_sides(
  %sri: !mo.tensor<[16, 64, 128, 128], f32>
) -> !mo.tensor<[1, 1024, 128, 128], f32> {
  %pc = mo.constant <#M.dense_array<1.0> : !M.array<1xf32>> : !mo.tensor<[], f32>

  %b0 = mo.static.broadcast_to(%pc) : !mo.tensor<[], f32> -> !mo.tensor<[1024], f32>
  %b1 = mo.static.reshape(%b0) : !mo.tensor<[1024], f32> -> !mo.tensor<[1, 1024, 1, 1], f32>

  %lhs = mo.static.reshape(%sri) : !mo.tensor<[16, 64, 128, 128], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>
  %pr = mo.static.broadcast_to(%b1) : !mo.tensor<[1, 1024, 1, 1], f32> -> !mo.tensor<[1, 1024, 1, 16384], f32>
  %rhs = mo.static.reshape(%pr) : !mo.tensor<[1, 1024, 1, 16384], f32> -> !mo.tensor<[1, 1024, 128, 128], f32>

  %out = mo.sub(%lhs, %rhs) : !mo.tensor<[1, 1024, 128, 128], f32>

  mo.output %out : !mo.tensor<[1, 1024, 128, 128], f32>
}

// The initial implementation of this optimization could loop forever with
// BroadcastReshapePattern on this case. The reshapes fold away.
//
// CHECK-LABEL: mo.graph
// CHECK-SAME: @minimal([[VARG0:[^ ]*]]: !mo.tensor<[300, 1], f32>) -> !mo.tensor<[300, 1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[], f32> -> !mo.tensor<[300, 1], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.min([[VARG0]], [[V1]]) : !mo.tensor<[300, 1], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.relu([[V2]]) : !mo.tensor<[300, 1], f32>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[300, 1], f32>
mo.graph @minimal(%arg0: !mo.tensor<[300, 1], f32>) -> !mo.tensor<[300, 1], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[], f32> -> !mo.tensor<[], f32>
  %2 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32> -> !mo.tensor<[300], f32>
  %3 = mo.static.reshape(%2) : !mo.tensor<[300], f32> -> !mo.tensor<[300, 1], f32>
  %4 = mo.min(%arg0, %3) : !mo.tensor<[300, 1], f32>
  %5 = mo.relu(%4) : !mo.tensor<[300, 1], f32>
  mo.output %5 : !mo.tensor<[300, 1], f32>
}

// On CPU, removing a trailing 1, adding a broadcast constant and putting the 1
// back used to loop between the reshape patterns; the add now moves to [8, 1].
// CHECK-LABEL: mo.graph @no_loop_squeeze_add_constant_unsqueeze
// CHECK:       %[[C:.*]] = mo.constant
// CHECK-NEXT:  %[[B:.*]] = mo.static.broadcast_to(%[[C]]) : !mo.tensor<[], f32> -> !mo.tensor<[8, 1], f32>
// CHECK-NEXT:  %[[A:.*]] = mo.add(%{{.*}}, %[[B]]) : !mo.tensor<[8, 1], f32>
mo.graph @no_loop_squeeze_add_constant_unsqueeze(%x: !mo.tensor<[8, 1], f32, cpu:0>) -> !mo.tensor<[8, 1], f32, cpu:0> {
  %c = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32, cpu:0>
  %cb = mo.static.broadcast_to(%c) : !mo.tensor<[], f32, cpu:0> -> !mo.tensor<[8], f32, cpu:0>
  %r = mo.static.reshape(%x) : !mo.tensor<[8, 1], f32, cpu:0> -> !mo.tensor<[8], f32, cpu:0>
  %a = mo.add(%r, %cb) : !mo.tensor<[8], f32, cpu:0>
  %y = mo.static.reshape(%a) : !mo.tensor<[8], f32, cpu:0> -> !mo.tensor<[8, 1], f32, cpu:0>
  mo.output %y : !mo.tensor<[8, 1], f32, cpu:0>
}

// The view is the si32 exponent of a pow with a constant f32 base: the pow
// keeps its f32 result after the swim.
// CHECK-LABEL: mo.graph @swim_reshape_of_exponent_through_pow
// CHECK:       mo.pow({{.*}}) : !mo.tensor<[8, 1], f32>
mo.graph @swim_reshape_of_exponent_through_pow(%e: !mo.tensor<[8, 1], si32>) -> !mo.tensor<[8], f32> {
  %c = mo.constant <#M.dense_array<2.0> : tensor<f32>> : !mo.tensor<[], f32>
  %cb = mo.static.broadcast_to(%c) : !mo.tensor<[], f32> -> !mo.tensor<[8], f32>
  %r = mo.static.reshape(%e) : !mo.tensor<[8, 1], si32> -> !mo.tensor<[8], si32>
  %p = mo.pow(%cb : !mo.tensor<[8], f32>, %r : !mo.tensor<[8], si32>) : !mo.tensor<[8], f32>
  mo.output %p : !mo.tensor<[8], f32>
}
