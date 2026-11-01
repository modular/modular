// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'slice_shape_func_success':
// CHECK-NEXT: 'slice_shape_func_success' returned tensor<2xf32> [2.000000e+00, 3.000000e+00]
mo.graph @slice_shape_func_success() -> !mo.tensor<[D0], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %start = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %stop = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  %result = mo.slice<() -> D0>(%input, %start, %stop, %step) :
    (!mo.tensor<[5], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>)
    -> !mo.tensor<[D0], f32>

  mo.output %result : !mo.tensor<[D0], f32>
}

// CHECK-LABEL: --- Running 'range_shape_func_success':
// CHECK-NEXT: 'range_shape_func_success' returned tensor<5xsi32> [0, 2, 4, 6, 8]
mo.graph @range_shape_func_success() -> !mo.tensor<[D0], si32> {
  %start = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %stop = mo.constant <#M.dense_array<10> : tensor<si32>> : !mo.tensor<[], si32>
  %step = mo.constant <#M.dense_array<2> : tensor<si32>> : !mo.tensor<[], si32>

  %result = mo.range<() -> D0>(%start, %stop, %step) :
    (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32>

  mo.output %result : !mo.tensor<[D0], si32>
}

// CHECK-LABEL: --- Running 'gather_shape_func_success':
// CHECK-NEXT: 'gather_shape_func_success' returned tensor<2x2xf32> [1.000000e+00, 2.000000e+00, 5.000000e+00, 6.000000e+00]
mo.graph @gather_shape_func_success() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>> : !mo.tensor<[4, 2], f32>
  %indices = mo.constant <#M.dense_array<0, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  %result = mo.gather<() -> D0, D1>(%input, %indices) [axis = 0] :
    (!mo.tensor<[4, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[D0, D1], f32>

  mo.output %result : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: --- Running 'reshape_shape_func_success':
// CHECK-NEXT: 'reshape_shape_func_success' returned tensor<2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]
mo.graph @reshape_shape_func_success() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %target_shape = mo.constant <#M.dense_array<2, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  %result = mo.reshape<() -> D0, D1>(%input, %target_shape) :
    (!mo.tensor<[6], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[D0, D1], f32>

  mo.output %result : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: --- Running 'broadcast_to_shape_func_success':
// CHECK-NEXT: 'broadcast_to_shape_func_success' returned tensor<2x3xf32>
mo.graph @broadcast_to_shape_func_success() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<5.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %target_shape = mo.constant <#M.dense_array<2, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  %result = mo.broadcast_to<() -> D0, D1>(%input, %target_shape) :
    (!mo.tensor<[1], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[D0, D1], f32>

  mo.output %result : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: --- Running 'tile_shape_func_success':
// CHECK-NEXT: 'tile_shape_func_success' returned tensor<4x6xf32>
mo.graph @tile_shape_func_success() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %repeats = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %result = mo.tile<() -> D0, D1>(%input, %repeats) :
    (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, D1], f32>

  mo.output %result : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: --- Running 'pad_shape_func_success':
// CHECK-NEXT: 'pad_shape_func_success' returned tensor<5x7xf32>
mo.graph @pad_shape_func_success() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %paddings = mo.constant <#M.dense_array<1, 2, 1, 3> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %pad_value = mo.constant <#M.dense_array<0.0> : tensor<1xf32>> : !mo.tensor<[], f32>

  %result = mo.pad.constant<() -> D0, D1>(%input, %paddings, %pad_value) :
    (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[D0, D1], f32>

  mo.output %result : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: --- Running 'squeeze_shape_func_success':
// CHECK-NEXT: 'squeeze_shape_func_success' returned tensor<2xsi64>
mo.graph @squeeze_shape_func_success() -> !mo.tensor<[D0], si64> {
  %input_shape = mo.constant <#M.dense_array<2, 1, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %indices = mo.constant <#M.dense_array<1, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %result = mo.squeeze_shape<() -> D0>(%input_shape, %indices) :
    (!mo.tensor<[4], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0], si64>

  mo.output %result : !mo.tensor<[D0], si64>
}

// CHECK-LABEL: --- Running 'unsqueeze_shape_func_success':
// CHECK-NEXT: 'unsqueeze_shape_func_success' returned tensor<5xsi64>
mo.graph @unsqueeze_shape_func_success() -> !mo.tensor<[D0], si64> {
  %input_shape = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %indices = mo.constant <#M.dense_array<0, 2, 4> : tensor<3xsi64>> : !mo.tensor<[3], si64>

  %result = mo.unsqueeze_shape<() -> D0>(%input_shape, %indices) :
    (!mo.tensor<[2], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[D0], si64>

  mo.output %result : !mo.tensor<[D0], si64>
}

// CHECK-LABEL: --- Running 'reduce_add_shape_func_success':
// CHECK-NEXT: 'reduce_add_shape_func_success' returned tensor<2x1xf32>
mo.graph @reduce_add_shape_func_success() -> !mo.tensor<[2, 1], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %result = mo.reduce.add(%input) :
    (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>

  mo.output %result : !mo.tensor<[2, 1], f32>
}
