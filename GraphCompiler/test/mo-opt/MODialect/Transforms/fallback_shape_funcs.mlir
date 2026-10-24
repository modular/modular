// UNSUPPORTED: ubsan

// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-add-fallback-shape-funcs | FileCheck %s

mo.graph @avg_pool_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.avg_pool<() -> D4, D5, D6, D7>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings, count_boundary = false] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @avg_pool_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER_SHAPE:arg.]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.avg_pool"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER_SHAPE]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]] :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.avg_pool(%[[INPUT]]) [filter_shape = %[[FILTER_SHAPE]], strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], count_boundary = false]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @avg_pool_ceil_mode_true_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.avg_pool_ceil_mode_true<() -> D4, D5, D6, D7>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings, count_boundary = false] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @avg_pool_ceil_mode_true_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER_SHAPE:arg.]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.avg_pool_ceil_mode_true"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER_SHAPE]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]] :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.avg_pool_ceil_mode_true(%[[INPUT]]) [filter_shape = %[[FILTER_SHAPE]], strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], count_boundary = false]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @batch_matmul<D0, D1, D2, D3, D4, D5 -> D6, D7, D8>(
  %lhs: !mo.tensor<[D0, D1, D2], f32>,
  %rhs: !mo.tensor<[D3, D4, D5], f32>
) -> !mo.tensor<[D6, D7, D8], f32> {

  %res = mo.batch_matmul<() -> D9, D10, D11>(%lhs, %rhs) :
    (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[D9, D10, D11], f32>

  mo.output<D9, D10, D11> %res : !mo.tensor<[D9, D10, D11], f32>
}
// CHECK-LABEL: mo.graph @batch_matmul
// CHECK-SAME: <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[D6:.*]], [[D7:.*]], [[D8:.*]]>(
// CHECK-SAME:   %[[LHS:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>,
// CHECK-SAME:   %[[RHS:.*]]: !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D6]], [[D7]], [[D8]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["mo.batch_matmul"]<() -> [[D9:.*]], [[D10:.*]], [[D11:.*]]>(%[[LHS]], %[[RHS]])
// CHECK-SAME:     [dataDeptTensors = []]
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.batch_matmul(%[[LHS]], %[[RHS]]) :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]], [[D2]]], f32>, !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>) -> !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>
//
// CHECK-NEXT:   mo.output<[[D9]], [[D10]], [[D11]]> %[[RES]] : !mo.tensor<[[[D9]], [[D10]], [[D11]]], f32>


mo.graph @max_pool_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.max_pool<() -> D4, D5, D6, D7>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @max_pool_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER_SHAPE:arg.]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.max_pool"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER_SHAPE]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]] :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.max_pool(%[[INPUT]]) [filter_shape = %[[FILTER_SHAPE]], strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]]]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @max_pool_ceil_mode_true_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter_shape: !mo.tensor<[2], si64>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.max_pool_ceil_mode_true<() -> D4, D5, D6, D7>(%input) [filter_shape = %filter_shape, strides = %strides, dilations = %dilations, paddings = %paddings] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @max_pool_ceil_mode_true_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER_SHAPE:arg.]]: !mo.tensor<[2], si64>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.max_pool_ceil_mode_true"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER_SHAPE]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]] :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.max_pool_ceil_mode_true(%[[INPUT]]) [filter_shape = %[[FILTER_SHAPE]], strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]]]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @layer_norm<D0, D1 -> D2, D3>(
  %input: !mo.tensor<[D0, D1], f32>,
  %gamma: !mo.tensor<[], f32>,
  %beta: !mo.tensor<[], f32>,
  %epsilon: !mo.tensor<[], f32>
) -> !mo.tensor<[D2, D3], f32> {

  %res = mo.reduce.layer_norm<() -> D4, D5>(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D4, D5], f32>

  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
// CHECK-LABEL: mo.graph @layer_norm
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>,
// CHECK-SAME:   %[[GAMMA:arg1]]: !mo.tensor<[], f32>,
// CHECK-SAME:   %[[BETA:arg2]]: !mo.tensor<[], f32>,
// CHECK-SAME:   %[[EPSILON:.*]]: !mo.tensor<[], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["mo.reduce.layer_norm"]<() -> [[D4:.*]], [[D5:.*]]>(
// CHECK-SAME:     %[[INPUT]], %[[GAMMA]], %[[BETA]], %[[EPSILON]]) [dataDeptTensors = []] :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.reduce.layer_norm(%[[INPUT]], %[[GAMMA]], %[[BETA]], %[[EPSILON]]) :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
// CHECK-SAME:     -> !mo.tensor<[[[D4]], [[D5]]], f32>
//
// CHECK-NEXT:   mo.output<[[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]]], f32>



mo.graph @broadcast_to_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[1, 3], f32>,
  %target_shape: !mo.tensor<[3], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.broadcast_to<() -> D3, D4, D5>(%input, %target_shape) :
    (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @broadcast_to_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 3], f32>,
// CHECK-SAME:   %[[TARGET_SHAPE:.*]]: !mo.tensor<[3], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.broadcast_to"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[TARGET_SHAPE]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32]] :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.broadcast_to(%[[INPUT]], %[[TARGET_SHAPE]]) :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>


mo.graph @broadcast_shape_fallback_shape_func<() -> D0>(
  %lhs: !mo.tensor<[1], si32>,
  %rhs: !mo.tensor<[2], si32>
) -> !mo.tensor<[D0], si32> {

  %res = mo.broadcast_shape<() -> D1>(%lhs, %rhs) :
    (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[D1], si32>

  mo.output<D1> %res : !mo.tensor<[D1], si32>
}
// CHECK-LABEL: mo.graph @broadcast_shape_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]]>(
// CHECK-SAME:   %[[LHS:.*]]: !mo.tensor<[1], si32>,
// CHECK-SAME:   %[[RHS:.*]]: !mo.tensor<[2], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]]], si32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.broadcast_shape"]<() -> [[D1:.*]]>(%[[LHS]], %[[RHS]])
// CHECK-SAME:   [dataDeptTensors = []] :
// CHECK-SAME:   (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.broadcast_shape(%[[LHS]], %[[RHS]]) :
// CHECK-SAME:   (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D1]]], si32>
//
// CHECK-NEXT: mo.output<[[D1]]> %[[RES]] : !mo.tensor<[[[D1]]], si32>


mo.graph @concat_variadic_input<D0, D1, D2, D3 -> D4, D5>(
  %input1: !mo.tensor<[D0, D1], f32>,
  %input2: !mo.tensor<[D2, D3], f32>
) -> !mo.tensor<[D4, D5], f32> {

  %res = mo.concat<() -> D6, D7>[1] (%input1, %input2) :
    (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D6, D7], f32>

  mo.output<D6, D7> %res : !mo.tensor<[D6, D7], f32>
}
// CHECK-LABEL: mo.graph @concat_variadic_input
// CHECK-SAME: <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]] -> [[D4:.*]], [[D5:.*]]>(
// CHECK-SAME:   %[[INPUT1:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>,
// CHECK-SAME:   %[[INPUT2:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D4]], [[D5]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["mo.concat"]<() -> [[D6:.*]], [[D7:.*]]>(%[[INPUT1]], %[[INPUT2]])
// CHECK-SAME:     [kgenParams = {{.*}}axis = 1 : index{{.*}}input_rank = 2 : index{{.*}}input_type = f32{{.*}}, dataDeptTensors = []] :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.concat[1] (%[[INPUT1]], %[[INPUT2]]) :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[[[D6]], [[D7]]], f32>
//
// CHECK-NEXT:   mo.output<[[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D6]], [[D7]]], f32>



mo.graph @conv_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter: !mo.tensor<[2, 2, 3, 1], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>,
  %num_groups: !mo.tensor<[], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.conv<() -> D4, D5, D6, D7>(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, num_groups = %num_groups, filter_layout = ""] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @conv_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER:.*]]: !mo.tensor<[2, 2, 3, 1], f32>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>, %[[NUM_GROUPS:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.conv"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]], %[[NUM_GROUPS]])
// CHECK-SAME:   [dataDeptTensors = [2 : i32, 3 : i32, 4 : i32, 5 : i32]] :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], num_groups = %[[NUM_GROUPS]], filter_layout = ""]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @conv_transpose_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter: !mo.tensor<[2, 2, 3, 1], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>,
  %out_paddings: !mo.tensor<[2], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.conv_transpose<() -> D4, D5, D6, D7>(%input, %filter)
    [strides = %strides, dilations = %dilations, paddings = %paddings, output_paddings = %out_paddings, filter_layout = ""] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @conv_transpose_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[32, 10, 20, 3], f32>, %[[FILTER:.*]]: !mo.tensor<[2, 2, 3, 1], f32>, %[[STRIDES:.*]]: !mo.tensor<[2], si64>, %[[DILATIONS:.*]]: !mo.tensor<[2], si64>, %[[PADDINGS:.*]]: !mo.tensor<[4], si64>, %[[OUTPUT_PADDINGS:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.conv_transpose"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[FILTER]], %[[STRIDES]], %[[DILATIONS]], %[[PADDINGS]], %[[OUTPUT_PADDINGS]])
// CHECK-SAME:   [dataDeptTensors = [2 : i32, 3 : i32, 4 : i32, 5 : i32]] :
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.conv_transpose(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], output_paddings = %[[OUTPUT_PADDINGS]], filter_layout = ""]
// CHECK-SAME:   (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @gather_fallback_shape_func<D0, D1 -> D2, D3, D4>(
  %input : !mo.tensor<[10, D0], f32>,
  %indices : !mo.tensor<[D1, 20], si32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %res = mo.gather<() -> D5, D6, D7>(%input, %indices) [axis = 0] :
    (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 20], si32>) -> !mo.tensor<[D5, D6, D7], f32>

  mo.output<D5, D6, D7> %res : !mo.tensor<[D5, D6, D7], f32>
}
// CHECK-LABEL: mo.graph @gather_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]], [[D4:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[10, [[D0]]], f32>
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[[[D1]], 20], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]], [[D4]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.gather"]<() -> [[D5:.*]], [[D6:.*]], [[D7:.*]]>(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   [kgenParams = {axis = 0 : index}, dataDeptTensors = []] :
// CHECK-SAME:   (!mo.tensor<[10, [[D0]]], f32>, !mo.tensor<[[[D1]], 20], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.gather(%[[INPUT]], %[[INDICES]]) [axis = 0] :
// CHECK-SAME:   (!mo.tensor<[10, [[D0]]], f32>, !mo.tensor<[[[D1]], 20], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D5]], [[D6]], [[D7]]], f32>


mo.graph @gather_nd_fallback_shape_func<D0, D1 -> D2, D3>(
  %input : !mo.tensor<[D0, D1, 20], f32>,
  %indices : !mo.tensor<[D0, 1], si32>
) -> !mo.tensor<[D2, D3], f32> {

  %res = mo.gather_nd<() -> D4, D5>(%input, %indices)
    [batchDims = 1] :
    (!mo.tensor<[D0, D1, 20], f32>, !mo.tensor<[D0, 1], si32>) -> !mo.tensor<[D4, D5], f32>

  mo.output<D4, D5> %res : !mo.tensor<[D4, D5], f32>
}
// CHECK-LABEL: mo.graph @gather_nd_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]], 20], f32>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[[[D0]], 1], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.gather_nd"]<() -> [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   [kgenParams = {batch_dims = 1 : index}, dataDeptTensors = []] :
// CHECK-SAME:   (!mo.tensor<[[[D0]], [[D1]], 20], f32>, !mo.tensor<[[[D0]], 1], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.gather_nd(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   [batchDims = 1] :
// CHECK-SAME:   (!mo.tensor<[[[D0]], [[D1]], 20], f32>, !mo.tensor<[[[D0]], 1], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]]], f32>


mo.graph @pad_constant_fallback_shape_func<() -> D0, D1, D2>(
  %input : !mo.tensor<[1, 2, 3], f32>,
  %paddings : !mo.tensor<[6], si32>,
  %scalar: !mo.tensor<[], f32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.pad.constant<() -> D3, D4, D5>(%input, %paddings, %scalar) :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @pad_constant_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[6], si32>,
// CHECK-SAME:   %[[SCALAR:.*]]: !mo.tensor<[], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.pad.constant"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[PADDINGS]], %[[SCALAR]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32]] :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>, !mo.tensor<[], f32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.pad.constant(%[[INPUT]], %[[PADDINGS]], %[[SCALAR]]) :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>, !mo.tensor<[], f32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>

mo.graph @pad_reflect_fallback_shape_func<() -> D0, D1, D2>(
  %input : !mo.tensor<[1, 2, 3], f32>,
  %paddings : !mo.tensor<[6], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.pad.reflect<() -> D3, D4, D5>(%input, %paddings) :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @pad_reflect_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[6], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.pad.reflect"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[PADDINGS]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32]] :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.pad.reflect(%[[INPUT]], %[[PADDINGS]]) :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>

mo.graph @pad_repeat_fallback_shape_func<() -> D0, D1, D2>(
  %input : !mo.tensor<[1, 2, 3], f32>,
  %paddings : !mo.tensor<[6], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.pad.repeat<() -> D3, D4, D5>(%input, %paddings) :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @pad_repeat_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME:   %[[PADDINGS:.*]]: !mo.tensor<[6], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.pad.repeat"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[PADDINGS]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32]] :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.pad.repeat(%[[INPUT]], %[[PADDINGS]]) :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[6], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>


mo.graph @range(%start: !mo.tensor<[], si32>, %stop: !mo.tensor<[], si32>, %step: !mo.tensor<[], si32>) -> !mo.tensor<[D1], si32> {
  %out = mo.range<() -> D1>(%start, %stop, %step) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D1], si32>
  mo.output %out: !mo.tensor<[D1], si32>
}
// CHECK-LABEL: mo.graph @range
// CHECK:(%[[START:.*]]: !mo.tensor<[], si32>, %[[STOP:.*]]: !mo.tensor<[], si32>, %[[STEP:.*]]: !mo.tensor<[], si32>)
// CHECK-NEXT: mo.invoke_shape_func["mo.range"]<() -> [[RANGE_PARAM:.*]]>(%[[START]], %[[STOP]], %[[STEP]]) [dataDeptTensors = [0 : i32, 1 : i32, 2 : i32]] : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>)
// CHECK-NEXT: mo.range(%[[START]], %[[STOP]], %[[STEP]]) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[[[RANGE_PARAM]]], si32>
mo.graph @reshape_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[1, 3], f32>,
  %target_shape: !mo.tensor<[3], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.reshape<() -> D3, D4, D5>(%input, %target_shape) :
    (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @reshape_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[1, 3], f32>,
// CHECK-SAME:   %[[TARGET_SHAPE:.*]]: !mo.tensor<[3], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.reshape"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[TARGET_SHAPE]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32]] :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.reshape(%[[INPUT]], %[[TARGET_SHAPE]]) :
// CHECK-SAME:   (!mo.tensor<[1, 3], f32>, !mo.tensor<[3], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>


mo.graph @scatter_fallback_shape_func<D0, D1, D2, D3, D4, D5 -> D6, D7>(
  %input: !mo.tensor<[D0, D1], f32>,
  %updates: !mo.tensor<[D2, D3], f32>,
  %indices: !mo.tensor<[D4, D5], si32>
) -> !mo.tensor<[D6, D7], f32> {

  %res = mo.scatter<() -> D8, D9>(%input, %updates, %indices) [axis = 0] : (
    !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>, !mo.tensor<[D4, D5], si32>
  ) -> !mo.tensor<[D8, D9], f32>

  mo.output<D8, D9> %res : !mo.tensor<[D8, D9], f32>
}
// CHECK-LABEL: mo.graph @scatter_fallback_shape_func
// CHECK-SAME:    <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>,
// CHECK-SAME:    %[[UPDATES:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>,
// CHECK-SAME:    %[[INDICES:.*]]: !mo.tensor<[[[D4]], [[D5]]], si32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D6]], [[D7]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["mo.scatter"]<() -> [[D8:.*]], [[D9:.*]]>(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
// CHECK-SAME:     [kgenParams = {axis = 0 : index}, dataDeptTensors = []] :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.scatter(%[[INPUT]], %[[UPDATES]], %[[INDICES]]) [axis = 0] : (
// CHECK-SAME:     !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si32>
// CHECK-SAME:   ) -> !mo.tensor<[[[D8]], [[D9]]], f32>
//
// CHECK-NEXT:   mo.output<[[D8]], [[D9]]> %[[RES]] : !mo.tensor<[[[D8]], [[D9]]], f32>
// CHECK-NEXT: }


mo.graph @scatter_nd_fallback_shape_func<D0, D1, D2, D3, D4, D5 -> D6, D7>(
  %input: !mo.tensor<[D0, D1], f32>,
  %updates: !mo.tensor<[D2, D3], f32>,
  %indices: !mo.tensor<[D4, D5], si32>
) -> !mo.tensor<[D6, D7], f32> {

  %res = mo.scatter_nd<() -> D8, D9>(%input, %updates, %indices) : (
    !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>, !mo.tensor<[D4, D5], si32>
  ) -> !mo.tensor<[D8, D9], f32>

  mo.output<D8, D9> %res : !mo.tensor<[D8, D9], f32>
}
// CHECK-LABEL: mo.graph @scatter_nd_fallback_shape_func
// CHECK-SAME:    <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]] -> [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>,
// CHECK-SAME:    %[[UPDATES:.*]]: !mo.tensor<[[[D2]], [[D3]]], f32>,
// CHECK-SAME:    %[[INDICES:.*]]: !mo.tensor<[[[D4]], [[D5]]], si32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D6]], [[D7]]], f32> {
//
// CHECK-NEXT:   mo.invoke_shape_func["mo.scatter_nd"]<() -> [[D8:.*]], [[D9:.*]]>(%[[INPUT]], %[[UPDATES]], %[[INDICES]])
// CHECK-SAME:     [dataDeptTensors = []] :
// CHECK-SAME:     (!mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si32>)
//
// CHECK-NEXT:   %[[RES:.*]] = mo.scatter_nd(%[[INPUT]], %[[UPDATES]], %[[INDICES]]) : (
// CHECK-SAME:     !mo.tensor<[[[D0]], [[D1]]], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]]], si32>
// CHECK-SAME:   ) -> !mo.tensor<[[[D8]], [[D9]]], f32>
//
// CHECK-NEXT:   mo.output<[[D8]], [[D9]]> %[[RES]] : !mo.tensor<[[[D8]], [[D9]]], f32>
// CHECK-NEXT: }


mo.graph @slice_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[1, 2, 3], f32>,
  %start: !mo.tensor<[3], si32>,
  %stop: !mo.tensor<[3], si32>,
  %step: !mo.tensor<[3], si32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  %res = mo.slice<() -> D3, D4, D5>(%input, %start, %stop, %step) :
    (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
    -> !mo.tensor<[D3, D4, D5], f32>

  mo.output<D3, D4, D5> %res : !mo.tensor<[D3, D4, D5], f32>
}
// CHECK-LABEL: mo.graph @slice_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[1, 2, 3], f32>,
// CHECK-SAME: %[[START:arg1]]: !mo.tensor<[3], si32>,
// CHECK-SAME: %[[STOP:arg2]]: !mo.tensor<[3], si32>,
// CHECK-SAME: %[[STEP:.*]]: !mo.tensor<[3], si32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]], [[D2]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.slice"]<() -> [[D3:.*]], [[D4:.*]], [[D5:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[START]], %[[STOP]], %[[STEP]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32]] :
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[START]], %[[STOP]], %[[STEP]])
// CHECK-SAME:   (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>)
// CHECK-SAME:   -> !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
//
// CHECK-NEXT: mo.output<[[D3]], [[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]], [[D5]]], f32>
mo.graph @squeeze_shape_fallback_shape_func<D0, D1 -> D2>(
  %input:   !mo.tensor<[D0], si64>,
  %indices: !mo.tensor<[D1], si64>
) -> !mo.tensor<[D2], si64> {

  %res = mo.squeeze_shape<() -> D3>(%input, %indices) :
    (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>

  mo.output<D3> %res : !mo.tensor<[D3], si64>
}
// CHECK-LABEL: mo.graph @squeeze_shape_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]]], si64>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[[[D1]]], si64>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]]], si64> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.squeeze_shape"]<() -> [[D3:.*]]>(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   [dataDeptTensors = []] : (!mo.tensor<[[[D0]]], si64>, !mo.tensor<[[[D1]]], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.squeeze_shape(%[[INPUT]], %[[INDICES]]) :
// CHECK-SAME:   (!mo.tensor<[[[D0]]], si64>, !mo.tensor<[[[D1]]], si64>) -> !mo.tensor<[[[D3]]], si64>
//
// CHECK-NEXT: mo.output<[[D3]]> %[[RES]] : !mo.tensor<[[[D3]]], si64>


mo.graph @transpose<D1, D2 -> D3, D4>(%arg0: !mo.tensor<[D1, D2], f32>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32> {
  %0 = mo.rebind<() -> D5, D6>(%arg0) : !mo.tensor<[D1, D2], f32> -> !mo.tensor<[D5, D6], f32>
  %1 = mo.transpose<() -> D7, D8>(%0, %arg1) : (!mo.tensor<[D5, D6], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D7, D8], f32>
  %2 = mo.rebind(%1) : !mo.tensor<[D7, D8], f32> -> !mo.tensor<[D3, D4], f32>
  mo.output<D7, D8> %1 : !mo.tensor<[D7, D8], f32>
}
// CHECK-LABEL: mo.graph @transpose
// CHECK: (%[[I0:.*]]: !mo.tensor<[D1, D2], f32>, %[[I1:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32>
// CHECK-NEXT: %[[RB:.*]] = mo.rebind<() -> D5, D6>(%arg0) : !mo.tensor<[D1, D2], f32> -> !mo.tensor<[D5, D6], f32>
// CHECK-NEXT: mo.invoke_shape_func["mo.transpose"]<() -> D7, D8>(%[[RB]], %[[I1]]) [dataDeptTensors = [1 : i32]] : (!mo.tensor<[D5, D6], f32>, !mo.tensor<[2], si64>)
// CHECK-NEXT: %[[TRANS:.*]] = mo.transpose(%[[RB]], %[[I1]]) : (!mo.tensor<[D5, D6], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D7, D8], f32>
// CHECK: mo.output<D7, D8> %[[TRANS]] : !mo.tensor<[D7, D8], f32>


mo.graph @arg_nonzero<D0 -> D1>(%arg0: !mo.tensor<[D0, 10], f32>) -> !mo.tensor<[D1, 2], si64> {
  %0 = mo.arg_nonzero<() -> D2>(%arg0) : (!mo.tensor<[D0, 10], f32>) -> !mo.tensor<[D2, 2], si64>
  mo.output<D2> %0 : !mo.tensor<[D2, 2], si64>
}
// CHECK-LABEL: mo.graph @arg_nonzero
// CHECK:(%[[IN:.*]]: !mo.tensor<[[[IN_PARAM:.*]], 10], f32>) -> !mo.tensor<[[[OUT_PARAM:.*]], 2], si64>
// CHECK-NEXT: mo.invoke_shape_func["mo.arg_nonzero"]<() -> [[WHERE_PARAM:.*]], [[UNUSED_PARAM:.*]]>(%[[IN]]) [dataDeptTensors = [0 : i32]]
// CHECK-NEXT: %[[WHERE:.*]] = mo.arg_nonzero(%[[IN]]) : (!mo.tensor<[[[IN_PARAM]], 10], f32>) -> !mo.tensor<[[[WHERE_PARAM]], 2], si64>
// CHECK-NEXT: mo.output<[[WHERE_PARAM]]> %[[WHERE]] : !mo.tensor<[[[WHERE_PARAM]], 2], si64>

// An off-host data-dependent input is copied to the host before the shape
// function reads it; the kernel itself still consumes the device original.
mo.graph @arg_nonzero_gpu<D0 -> D1>(%arg0: !mo.tensor<[D0, 10], f32, dev:0>) -> !mo.tensor<[D1, 2], si64, dev:0> {
  %0 = mo.arg_nonzero<() -> D2>(%arg0) : (!mo.tensor<[D0, 10], f32, dev:0>) -> !mo.tensor<[D2, 2], si64, dev:0>
  mo.output<D2> %0 : !mo.tensor<[D2, 2], si64, dev:0>
}
// CHECK-LABEL: mo.graph @arg_nonzero_gpu
// CHECK: (%[[GPU_IN:.*]]: !mo.tensor<[{{.*}}, 10], f32, dev:0>) -> !mo.tensor<[{{.*}}, 2], si64, dev:0> {
// CHECK: %[[CHAIN:.*]] = mo.chain.create()
// CHECK: %[[GPU_HOST:.*]], %{{.*}} = mo.transfer[%[[CHAIN]]] %[[GPU_IN]] : !mo.tensor<[{{.*}}, 10], f32, dev:0> to <"cpu", 0>
// CHECK: mo.invoke_shape_func["mo.arg_nonzero"]<() -> {{.*}}>(%[[GPU_HOST]]) [dataDeptTensors = [0 : i32]]
// CHECK: %[[GPU_WHERE:.*]] = mo.arg_nonzero(%[[GPU_IN]]) : (!mo.tensor<[{{.*}}, 10], f32, dev:0>) -> !mo.tensor<[{{.*}}, 2], si64, dev:0>
// CHECK: mo.output<{{.*}}> %[[GPU_WHERE]] : !mo.tensor<[{{.*}}, 2], si64, dev:0>

mo.graph @top_k_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %K: !mo.tensor<[], si64>,
  %axis: !mo.tensor<[], si64>,
  %sorted : !mo.tensor<[], bool>
) -> (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D0, D1, D2, D3], si64>) {

  %res:2 = mo.top_k<() -> D8, D9, D10, D11>(%input, %K, %axis, %sorted) :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
    -> (!mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D8, D9, D10, D11], si64>)

  mo.output<D8, D9, D10, D11> %res#0, %res#1 : !mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D8, D9, D10, D11], si64>
}

// CHECK-LABEL: mo.graph @top_k_fallback_shape_func
// CHECK-NEXT:   mo.invoke_shape_func["mo.top_k"]<() -> [[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]]>({{.*}}) [dataDeptTensors = [1 : i32, 2 : i32]] : (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
// CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.top_k(%[[INPUT:.*]], %[[K:.*]], %[[AXIS:.*]], %[[SORTED:.*]]) : (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si64>)
// CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> %[[VALUES]], %[[INDICES]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si64>

mo.graph @bottom_k_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %K: !mo.tensor<[], si64>,
  %axis: !mo.tensor<[], si64>,
  %sorted : !mo.tensor<[], bool>
) -> (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D0, D1, D2, D3], si64>) {

  %res:2 = mo.bottom_k<() -> D8, D9, D10, D11>(%input, %K, %axis, %sorted) :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
    -> (!mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D8, D9, D10, D11], si64>)

  mo.output<D8, D9, D10, D11> %res#0, %res#1 : !mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D8, D9, D10, D11], si64>
}

// CHECK-LABEL: mo.graph @bottom_k_fallback_shape_func
// CHECK-NEXT:   mo.invoke_shape_func["mo.bottom_k"]<() -> [[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]]>({{.*}}) [dataDeptTensors = [1 : i32, 2 : i32]] : (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
// CHECK-NEXT:   %[[VALUES:.*]], %[[INDICES:.*]] = mo.bottom_k(%[[INPUT:.*]], %[[K:.*]], %[[AXIS:.*]], %[[SORTED:.*]]) : (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si64>)
// CHECK-NEXT:   mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> %[[VALUES]], %[[INDICES]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>, !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], si64>

mo.graph @tile_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[2, 3, 7, 11], f32>,
  %repeats: !mo.tensor<[4], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {

  %res = mo.tile<() -> D4, D5, D6, D7>(%input, %repeats) :
    (!mo.tensor<[2, 3, 7, 11], f32>, !mo.tensor<[4], si64>)
    -> !mo.tensor<[D4, D5, D6, D7], f32>

  mo.output<D4, D5, D6, D7> %res : !mo.tensor<[D4, D5, D6, D7], f32>
}

// CHECK-LABEL: mo.graph @tile_fallback_shape_func
// CHECK-SAME: <() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT:.*]]: !mo.tensor<[2, 3, 7, 11], f32>, %[[REPEATS:.*]]: !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.tile"]<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:   %[[INPUT]], %[[REPEATS]])
// CHECK-SAME:   [dataDeptTensors = [1 : i32]] :
// CHECK-SAME:   (!mo.tensor<[2, 3, 7, 11], f32>, !mo.tensor<[4], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.tile(%[[INPUT]], %[[REPEATS]])
// CHECK-SAME:   (!mo.tensor<[2, 3, 7, 11], f32>, !mo.tensor<[4], si64>)
// CHECK-SAME:   -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>
//
// CHECK-NEXT: mo.output<[[D4]], [[D5]], [[D6]], [[D7]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32>


mo.graph @roi_align_fallback_shape_func<D0, D1, D2, D3 -> D4, D5, D6, D7>(
  %input: !mo.tensor<[D0, D1, D2, D3], f32>
) -> !mo.tensor<[D4, D5, D6, D7], f32> {

  %output_height = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[], si64>
  %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %rois = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 2.0, 2.0> : tensor<5xf32>> : !mo.tensor<[1, 5], f32>

  %output = mo.roi_align<() -> D8, D9, D10, D11>(%input, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio)
    [aligned = false, mode = "AVG"]
    : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>

  mo.output<D8, D9, D10, D11> %output : !mo.tensor<[D8, D9, D10, D11], f32>
}

// CHECK-LABEL: mo.graph @roi_align_fallback_shape_func
// CHECK-SAME:  <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]] -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//
// CHECK-NEXT:  %[[OUTPUT_HEIGHT:.*]] = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[], si64>
// CHECK-NEXT:  %[[SPATIAL_SCALE:.*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT:  %[[SAMPLING_RATIO:.*]] = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT:  %[[ROIS:.*]] = mo.constant <#M.dense_array<0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00> : tensor<5xf32>> : !mo.tensor<[1, 5], f32>
//
// CHECK-NEXT:  mo.invoke_shape_func["mo.roi_align"]<() -> [[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]]>(
// CHECK-SAME:    %[[INPUT]], %[[ROIS]], %[[OUTPUT_HEIGHT]], %[[OUTPUT_HEIGHT]], %[[SPATIAL_SCALE]], %[[SAMPLING_RATIO]])
// CHECK-SAME:    [dataDeptTensors = [2 : i32, 3 : i32]]
// CHECK-SAME:    : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
//
// CHECK-NEXT:  %[[OUTPUT:.*]] = mo.roi_align(%[[INPUT]], %[[ROIS]], %[[OUTPUT_HEIGHT]], %[[OUTPUT_HEIGHT]], %[[SPATIAL_SCALE]], %[[SAMPLING_RATIO]])
// CHECK-SAME:    [aligned = false, mode = "AVG"]
// CHECK-SAME:    : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//
// CHECK-NEXT:  mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> %[[OUTPUT]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>


// GEX-4020 regression: output_height/output_width are block arguments here
// (not mo.constant), the shape a runtime-operand eager graph produces. The
// fallback previously truncated to the first four operands, dropping
// spatial_scale/sampling_ratio, which the registered shape function still
// declares -- so the emitted invoke_shape_func must carry all six.
mo.graph @roi_align_dynamic_output_size_fallback_shape_func<D0, D1, D2, D3 -> D4, D5, D6, D7>(
  %input: !mo.tensor<[D0, D1, D2, D3], f32>,
  %output_height: !mo.tensor<[], si64>,
  %output_width: !mo.tensor<[], si64>
) -> !mo.tensor<[D4, D5, D6, D7], f32> {

  %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %rois = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 2.0, 2.0> : tensor<5xf32>> : !mo.tensor<[1, 5], f32>

  %output = mo.roi_align<() -> D8, D9, D10, D11>(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio)
    [aligned = false, mode = "AVG"]
    : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D8, D9, D10, D11], f32>

  mo.output<D8, D9, D10, D11> %output : !mo.tensor<[D8, D9, D10, D11], f32>
}

// CHECK-LABEL: mo.graph @roi_align_dynamic_output_size_fallback_shape_func
// CHECK-SAME:  <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]] -> [[D4:.*]], [[D5:.*]], [[D6:.*]], [[D7:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>,
// CHECK-SAME:    %[[OUTPUT_HEIGHT:.*]]: !mo.tensor<[], si64>,
// CHECK-SAME:    %[[OUTPUT_WIDTH:.*]]: !mo.tensor<[], si64>
// CHECK-SAME:  ) -> !mo.tensor<[[[D4]], [[D5]], [[D6]], [[D7]]], f32> {
//
// CHECK-NEXT:  %[[SPATIAL_SCALE:.*]] = mo.constant <#M.dense_array<1.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT:  %[[SAMPLING_RATIO:.*]] = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT:  %[[ROIS:.*]] = mo.constant <#M.dense_array<0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00> : tensor<5xf32>> : !mo.tensor<[1, 5], f32>
//
// CHECK-NEXT:  mo.invoke_shape_func["mo.roi_align"]<() -> [[D8:.*]], [[D9:.*]], [[D10:.*]], [[D11:.*]]>(
// CHECK-SAME:    %[[INPUT]], %[[ROIS]], %[[OUTPUT_HEIGHT]], %[[OUTPUT_WIDTH]], %[[SPATIAL_SCALE]], %[[SAMPLING_RATIO]])
// CHECK-SAME:    [dataDeptTensors = [2 : i32, 3 : i32]]
// CHECK-SAME:    : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
//
// CHECK-NEXT:  %[[OUTPUT:.*]] = mo.roi_align(%[[INPUT]], %[[ROIS]], %[[OUTPUT_HEIGHT]], %[[OUTPUT_WIDTH]], %[[SPATIAL_SCALE]], %[[SAMPLING_RATIO]])
// CHECK-SAME:    [aligned = false, mode = "AVG"]
// CHECK-SAME:    : (!mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>
//
// CHECK-NEXT:  mo.output<[[D8]], [[D9]], [[D10]], [[D11]]> %[[OUTPUT]] : !mo.tensor<[[[D8]], [[D9]], [[D10]], [[D11]]], f32>


mo.graph @unsqueeze_shape_fallback_shape_func<D0, D1 -> D2>(
  %input:   !mo.tensor<[D0], si64>,
  %indices: !mo.tensor<[D1], si64>
) -> !mo.tensor<[D2], si64> {

  %res = mo.unsqueeze_shape<() -> D3>(%input, %indices) :
    (!mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>) -> !mo.tensor<[D3], si64>

  mo.output<D3> %res : !mo.tensor<[D3], si64>
}
// CHECK-LABEL: mo.graph @unsqueeze_shape_fallback_shape_func
// CHECK-SAME: <[[D0:.*]], [[D1:.*]] -> [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]]], si64>,
// CHECK-SAME:   %[[INDICES:.*]]: !mo.tensor<[[[D1]]], si64>
// CHECK-SAME: ) -> !mo.tensor<[[[D2]]], si64> {
//
// CHECK-NEXT: mo.invoke_shape_func["mo.unsqueeze_shape"]<() -> [[D3:.*]]>(%[[INPUT]], %[[INDICES]])
// CHECK-SAME:   [dataDeptTensors = []] : (!mo.tensor<[[[D0]]], si64>, !mo.tensor<[[[D1]]], si64>)
//
// CHECK-NEXT: %[[RES:.*]] = mo.unsqueeze_shape(%[[INPUT]], %[[INDICES]]) :
// CHECK-SAME:   (!mo.tensor<[[[D0]]], si64>, !mo.tensor<[[[D1]]], si64>) -> !mo.tensor<[[[D3]]], si64>
//
// CHECK-NEXT: mo.output<[[D3]]> %[[RES]] : !mo.tensor<[[[D3]]], si64>


// Note how the second invocation does not redefine "I1" or "B"
// CHECK-LABEL: mo.graph @broadcast_partially_bound_params
// CHECK: mo.invoke_shape_func["mo.broadcast_to"]<() -> I1, I2, I3>({{.*}}) [dataDeptTensors = [1 : i32]] : (!mo.tensor<[A, B], f32>, !mo.tensor<[3], si32>)
// CHECK: mo.invoke_shape_func["mo.broadcast_to"]<() -> F1, F2, I4>({{.*}}) [dataDeptTensors = [1 : i32]] : (!mo.tensor<[A, B], f32>, !mo.tensor<[3], si32>)
mo.graph @broadcast_partially_bound_params<A, B, C -> O1, O2, O3, O4, O5, O6>(
  %input: !mo.tensor<[A, B], f32>,
  %target_shape: !mo.tensor<[3], si32>
) -> (!mo.tensor<[O1, O2, O3], f32>, !mo.tensor<[O4, O5, O6], f32>) {

  %res1 = mo.broadcast_to<() -> I1, I2, I3>(%input, %target_shape) :
    (!mo.tensor<[A, B], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[I1, I2, I3], f32>

  // NOTE: only declares some of the output parameters, reuses some from res1 and input
  %res2 = mo.broadcast_to<() -> I4>(%input, %target_shape) :
    (!mo.tensor<[A, B], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[I1, B, I4], f32>

  mo.output<I1, I2, I3, I1, B, I4> %res1, %res2 : !mo.tensor<[I1, I2, I3], f32>, !mo.tensor<[I1, B, I4], f32>
}

// Notice how the `test_flag` parameter is properly forwarded to the invoke shape func
// CHECK-LABEL: @custom_shape_func
// CHECK: mo.invoke_shape_func["custom_op_with_custom_parameter_shape_func"]
// CHECK-SAME: kgenParams = {output_idx = 0 : index, test_flag = 0 : index}
mo.graph @custom_shape_func<() -> O1, O2, O3, O4, O5>(%arg0: !mo.tensor<[5, 5, 5, 5, 5], f32>) -> !mo.tensor<[O1, O2, O3, O4, O5], f32> {
  %0 = mo.custom<() -> A, B, C, D, E> [symbol = "custom_op_with_custom_parameter_shape_func", device = #M.device_ref<"cpu", 0>, parameters = {test_flag = 0 : index}](%arg0) : (!mo.tensor<[5, 5, 5, 5, 5], f32>) -> !mo.tensor<[A, B, C, D, E], f32>
  mo.output<A, B, C, D, E> %0 : !mo.tensor<[A, B, C, D, E], f32>
}

// Only the `Some[TileTensorable]` operand is copied to the host.
// CHECK-LABEL: @custom_shape_func_mixed_bounds
// CHECK: (%[[X:.*]]: !mo.tensor<[{{.*}}, 8], f32, dev:0>, %[[Y:.*]]: !mo.tensor<[{{.*}}], f32, dev:0>, %[[N:.*]]: !mo.tensor<[], si32>)
// CHECK: %[[CHAIN:.*]] = mo.chain.create()
// CHECK: %[[Y_HOST:.*]], %{{.*}} = mo.transfer[%[[CHAIN]]] %[[Y]] : !mo.tensor<[{{.*}}], f32, dev:0> to <"cpu", 0>
// CHECK-NOT: mo.transfer
// CHECK: mo.invoke_shape_func["custom_op_with_mixed_bound_shape_func"]<() -> {{.*}}>(%[[X]], %[[Y_HOST]], %[[N]]) [kgenParams = {output_idx = 0 : index}, dataDeptTensors = [1 : i32, 2 : i32]]
// CHECK: mo.custom{{.*}}[symbol = "custom_op_with_mixed_bound_shape_func"{{.*}}](%[[X]], %[[Y]], %[[N]])
mo.graph @custom_shape_func_mixed_bounds<D0, D1 -> D2>(%x: !mo.tensor<[D0, 8], f32, dev:0>, %y: !mo.tensor<[D1], f32, dev:0>, %n: !mo.tensor<[], si32>) -> !mo.tensor<[D2, 8], f32, dev:0> {
  %0 = mo.custom<() -> R> [symbol = "custom_op_with_mixed_bound_shape_func", device = #M.device_ref<"dev", 0>](%x, %y, %n) : (!mo.tensor<[D0, 8], f32, dev:0>, !mo.tensor<[D1], f32, dev:0>, !mo.tensor<[], si32>) -> !mo.tensor<[R, 8], f32, dev:0>
  mo.output<R> %0 : !mo.tensor<[R, 8], f32, dev:0>
}

// A `Tensor`-bound `TensorTuple` stays on the device.
// CHECK-LABEL: @custom_shape_func_tensor_tuple
// CHECK: (%[[X:.*]]: !mo.tensor<[{{.*}}], f32, dev:0>)
// CHECK-NOT: mo.transfer
// CHECK: mo.invoke_shape_func["trait_tensor_tuple_shape_op"]<() -> {{.*}}>(%[[X]], %[[X]]) [kgenParams = {output_idx = 0 : index}, dataDeptTensors = []]
mo.graph @custom_shape_func_tensor_tuple<D0, D1 -> D2, D3>(%x: !mo.tensor<[D0, D1], f32, dev:0>) -> !mo.tensor<[D2, D3], f32, dev:0> {
  %0 = mo.custom<() -> R0, R1> [symbol = "trait_tensor_tuple_shape_op", device = #M.device_ref<"dev", 0>](%x, %x) : (!mo.tensor<[D0, D1], f32, dev:0>, !mo.tensor<[D0, D1], f32, dev:0>) -> !mo.tensor<[R0, R1], f32, dev:0>
  mo.output<R0, R1> %0 : !mo.tensor<[R0, R1], f32, dev:0>
}
