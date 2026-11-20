// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=zeros --result-output-style=full %t.mlir | FileCheck %s

mo.graph @pool(%input : !mo.tensor<[1, 5, 5, 1], f32>) -> (!mo.tensor<[1, 3, 3, 1], f32>) {
  %d = "mo.constant"() {value = #M.dense_array<0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.> : tensor<1x5x5x1xf32>} : () -> !mo.tensor<[1, 5, 5, 1], f32>
  %fs = "mo.constant"() {value = #M.dense_array<3, 3> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %v = "mo.add"(%input, %d) : (!mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[1, 5, 5, 1], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
  %output = mo.max_pool(%v) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], f32>
}
// CHECK{LITERAL}: 'pool' returned tensor<1x3x3x1xf32> [1.200000e+01, 1.300000e+01, 1.400000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]

mo.graph @strided_pool(%input : !mo.tensor<[1, 5, 5, 1], f32>) -> (!mo.tensor<[1, 2, 2, 1], f32>) {
  %d = "mo.constant"() {value = #M.dense_array<0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.> : tensor<1x5x5x1xf32>} : () -> !mo.tensor<[1, 5, 5, 1], f32>
  %fs = "mo.constant"() {value = #M.dense_array<3, 3> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %st = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %v = "mo.add"(%input, %d) : (!mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[1, 5, 5, 1], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
  %output = mo.avg_pool(%v) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa, count_boundary = false] : (
    !mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[1, 2, 2, 1], f32>
  mo.output %output : !mo.tensor<[1, 2, 2, 1], f32>
}
// CHECK{LITERAL}: 'strided_pool' returned tensor<1x2x2x1xf32> [6.000000e+00, 8.000000e+00, 1.600000e+01, 1.800000e+01]

mo.graph @avg_pool_count_boundary_false(%input : !mo.tensor<[1, 5, 5, 1], f32>) -> (!mo.tensor<[1, 3, 3, 1], f32>) {
  %d = "mo.constant"() {value = #M.dense_array<0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.> : tensor<1x5x5x1xf32>} : () -> !mo.tensor<[1, 5, 5, 1], f32>
  %fs = "mo.constant"() {value = #M.dense_array<3, 3> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %st = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<1, 1, 1, 1> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %v = "mo.add"(%input, %d) : (!mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[1, 5, 5, 1], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
  %output = mo.avg_pool(%v) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa, count_boundary = false] : (
    !mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], f32>
}
// CHECK: 'avg_pool_count_boundary_false' returned tensor<1x3x3x1xf32> [3.0{{.*}}e+00, 4.5{{.*}}e+00, 6.0{{.*}}e+00, 1.05{{.*}}e+01, 1.2{{.*}}e+01, 1.35{{.*}}e+01, 1.8{{.*}}e+01, 1.95{{.*}}e+01, 2.1{{.*}}e+01]

mo.graph @avg_pool_count_boundary_true(%input : !mo.tensor<[1, 5, 5, 1], f32>) -> (!mo.tensor<[1, 3, 3, 1], f32>) {
  %d = "mo.constant"() {value = #M.dense_array<0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.> : tensor<1x5x5x1xf32>} : () -> !mo.tensor<[1, 5, 5, 1], f32>
  %fs = "mo.constant"() {value = #M.dense_array<3, 3> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %st = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<1, 1, 1, 1> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %v = "mo.add"(%input, %d) : (!mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[1, 5, 5, 1], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
  %output = mo.avg_pool(%v) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa, count_boundary = true] : (
    !mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], f32>
}
// CHECK: 'avg_pool_count_boundary_true' returned tensor<1x3x3x1xf32> [1.3{{.*}}e+00, 3.0{{.*}}e+00, 2.6{{.*}}e+00, 7.0{{.*}}e+00, 1.2{{.*}}e+01, 9.0{{.*}}e+00, 8.0{{.*}}e+00, 1.3{{.*}}e+01, 9.3{{.*}}e+00]

mo.graph @dilated_pool(%input : !mo.tensor<[1, 5, 5, 1], f32>) -> (!mo.tensor<[1, 1, 1, 1], f32>) {
  %d = "mo.constant"() {value = #M.dense_array<0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.> : tensor<1x5x5x1xf32>} : () -> !mo.tensor<[1, 5, 5, 1], f32>
  %fs = "mo.constant"() {value = #M.dense_array<3, 3> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %v = "mo.add"(%input, %d) : (!mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[1, 5, 5, 1], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
  %output = mo.max_pool(%v) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[1, 1, 1, 1], f32>
  mo.output %output : !mo.tensor<[1, 1, 1, 1], f32>
}
// CHECK{LITERAL}: 'dilated_pool' returned tensor<1x1x1x1xf32> [2.400000e+01]

mo.graph @avg_pool_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %filter_size = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %strides = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %dilations = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>

  mo.invoke_shape_func["mo.avg_pool"]<() -> D4, D5, D6, D7>(%input, %filter_size, %strides, %dilations, %paddings)
    [kgenParams = {ceil_mode = false}, dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si32>,
     !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[4], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'avg_pool_fallback_shape_func' returned tensor<32x4x20x3xsi32>

mo.graph @avg_pool_ceil_mode_true_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %filter_size = mo.constant <#M.dense_array<3, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %strides = mo.constant <#M.dense_array<2, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %dilations = mo.constant <#M.dense_array<1, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>

  mo.invoke_shape_func["mo.avg_pool_ceil_mode_true"]<() -> D4, D5, D6, D7>(%input, %filter_size, %strides, %dilations, %paddings)
    [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si32>,
     !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[4], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'avg_pool_ceil_mode_true_fallback_shape_func' returned tensor<32x5x10x3xsi32>

mo.graph @max_pool_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %filter_size = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %strides = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %dilations = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>

  mo.invoke_shape_func["mo.max_pool"]<() -> D4, D5, D6, D7>(%input, %filter_size, %strides, %dilations, %paddings)
    [kgenParams = {ceil_mode = false}, dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4: i32]] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si32>,
     !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[4], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'max_pool_fallback_shape_func' returned tensor<32x4x20x3xsi32>

mo.graph @max_pool_ceil_mode_true_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %filter_size = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %strides = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %dilations = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>

  mo.invoke_shape_func["mo.max_pool_ceil_mode_true"]<() -> D4, D5, D6, D7>(%input, %filter_size, %strides, %dilations, %paddings)
    [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4: i32]] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si32>,
     !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[4], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'max_pool_ceil_mode_true_fallback_shape_func' returned tensor<32x5x20x3xsi32>

mo.graph @max_pool_fallback_shape_func_non_zero_padding<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %filter_size = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %strides = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %dilations = mo.constant <#M.dense_array<2, 1> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %paddings = mo.constant <#M.dense_array<1, 1, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>

  mo.invoke_shape_func["mo.max_pool"]<() -> D4, D5, D6, D7>(%input, %filter_size, %strides, %dilations, %paddings)
    [kgenParams = {ceil_mode = false}, dataDeptTensors = [1 : i32, 2 : i32, 3 : i32, 4 : i32]] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si32>,
     !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[4], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'max_pool_fallback_shape_func_non_zero_padding' returned tensor<32x5x20x3xsi32>


mo.graph @avg_pool_count_boundary_false_ceil_mode_true(%input : !mo.tensor<[1, 4, 4, 1], f32>) -> (!mo.tensor<[1, 2, 2, 1], f32>) {
  %d = "mo.constant"() {value = #M.dense_array<0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.> : tensor<1x4x4x1xf32>} : () -> !mo.tensor<[1, 4, 4, 1], f32>
  %fs = "mo.constant"() {value = #M.dense_array<3, 3> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %st = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %v = "mo.add"(%input, %d) : (!mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[1, 4, 4, 1], f32>) -> !mo.tensor<[1, 4, 4, 1], f32>
  %output = mo.avg_pool_ceil_mode_true(%v) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa, count_boundary = false] : (
    !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[1, 2, 2, 1], f32>
  mo.output %output : !mo.tensor<[1, 2, 2, 1], f32>
}
// CHECK: 'avg_pool_count_boundary_false_ceil_mode_true' returned tensor<1x2x2x1xf32> [5.0{{.*}}e+00, 6.5{{.*}}e+00, 1.1{{.*}}e+01, 1.25{{.*}}e+01]

mo.graph @avg_pool_count_boundary_true_ceil_mode_true(%input : !mo.tensor<[1, 4, 4, 1], f32>) -> (!mo.tensor<[1, 2, 2, 1], f32>) {
  %d = "mo.constant"() {value = #M.dense_array<0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.> : tensor<1x4x4x1xf32>} : () -> !mo.tensor<[1, 4, 4, 1], f32>
  %fs = "mo.constant"() {value = #M.dense_array<3, 3> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %st = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %v = "mo.add"(%input, %d) : (!mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[1, 4, 4, 1], f32>) -> !mo.tensor<[1, 4, 4, 1], f32>
  %output = mo.avg_pool_ceil_mode_true(%v) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa, count_boundary = true] : (
    !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[1, 2, 2, 1], f32>
  mo.output %output : !mo.tensor<[1, 2, 2, 1], f32>
}
// CHECK: 'avg_pool_count_boundary_true_ceil_mode_true' returned tensor<1x2x2x1xf32> [5.0{{.*}}e+00, 4.33{{.*}}e+00, 7.33{{.*}}e+00, 5.55{{.*}}e+00]
