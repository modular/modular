// RUN: mo-opt %s --gc-pipeline -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

mo.graph @conv_transpose_default_hyperparameters() -> (!mo.tensor<[1, 3, 3, 1], si64>) {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3> : tensor<4xsi64>> : !mo.tensor<[1, 2, 2, 1], si64>
  %filter = mo.constant <#M.dense_array<0, 1, 2, 3> : tensor<4xsi64>> : !mo.tensor<[2, 2, 1, 1], si64, {layout = #mo.layout<RSFC>}>
  %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 1], si64>, !mo.tensor<[2, 2, 1, 1], si64, {layout = #mo.layout<RSFC>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], si64>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], si64>
}
// CHECK{LITERAL}: 'conv_transpose_default_hyperparameters' returned tensor<1x3x3x1xsi64> [0, 0, 1, 0, 4, 6, 4, 12, 9]


mo.graph @conv_transpose_non_default_hyperparameters() -> (!mo.tensor<[1, 2, 4, 1], si64>) {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3> : tensor<4xsi64>> : !mo.tensor<[1, 2, 2, 1], si64>
  %filter = mo.constant <#M.dense_array<0, 1, 2, 3> : tensor<4xsi64>> : !mo.tensor<[2, 2, 1, 1], si64, {layout = #mo.layout<RSFC>}>
  %st = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<1, 0, 0, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 1], si64>, !mo.tensor<[2, 2, 1, 1], si64, {layout = #mo.layout<RSFC>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[1, 2, 4, 1], si64>
  mo.output %output : !mo.tensor<[1, 2, 4, 1], si64>
}
// CHECK{LITERAL}: 'conv_transpose_non_default_hyperparameters' returned tensor<1x2x4x1xsi64> [0, 0, 4, 0, 4, 0, 12, 0]


mo.graph @conv_transpose_non_one_channel() -> (!mo.tensor<[1, 1, 3, 3], si64>) {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3> : tensor<4xsi64>> : !mo.tensor<[1, 1, 2, 2], si64>
  %filter = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<12xsi64>> : !mo.tensor<[1, 2, 3, 2], si64, {layout = #mo.layout<RSFC>}>
  %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 1, 2, 2], si64>, !mo.tensor<[1, 2, 3, 2], si64, {layout = #mo.layout<RSFC>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[1, 1, 3, 3], si64>
  mo.output %output : !mo.tensor<[1, 1, 3, 3], si64>
}
// CHECK{LITERAL}: 'conv_transpose_non_one_channel' returned tensor<1x1x3x3xsi64> [1, 3, 5, 10, 22, 34, 33, 43, 53]


mo.graph @conv_transpose_fallback_padded_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 9], f32>,
  %filter: !mo.tensor<[2, 2, 3, 9], f32, {layout = #mo.layout<RSFC>}>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %st = mo.constant <#M.dense_array<2, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %di = mo.constant <#M.dense_array<3, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %pa = mo.constant <#M.dense_array<1, 1, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  mo.invoke_shape_func["mo.conv_transpose"]<() -> D4, D5, D6, D7>(%input, %filter, %st, %di, %pa, %op)
    [dataDeptTensors = [2 : i32, 3 : i32, 4: i32, 5: i32]] :
    (!mo.tensor<[32, 10, 20, 9], f32>, !mo.tensor<[2, 2, 3, 9], f32, {layout = #mo.layout<RSFC>}>,
     !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[4], si32>, !mo.tensor<[2], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'conv_transpose_fallback_padded_shape_func' returned tensor<32x20x60x3xsi32>


mo.graph @conv3d_transpose(%input: !mo.tensor<[1, 2, 2, 2, 1],si64>, %filter: !mo.tensor<[2, 2, 2, 1, 1],si64, {layout = #mo.layout<QRSFC>}>) -> (!mo.tensor<[1, 3, 3, 3, 1], si64>) {
  %st = mo.constant <#M.dense_array<1, 1 ,1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %di = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>> : !mo.tensor<[6], si64>
  %op = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 2, 1], si64>, !mo.tensor<[2, 2, 2, 1, 1], si64, {layout = #mo.layout<QRSFC>}>,
    !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[1, 3, 3, 3, 1], si64>
  mo.output %output : !mo.tensor<[1, 3, 3, 3, 1], si64>
}
// CHECK{LITERAL}: 'conv3d_transpose' returned tensor<1x3x3x3x1xsi64> [1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 4, 2, 4, 8, 4, 2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2, 1]

mo.graph @conv3d_transpose_const_filter(%input: !mo.tensor<[1, 2, 2, 2, 1],si64>) -> (!mo.tensor<[1, 3, 3, 3, 1], si64>) {
  %filter = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7> : tensor<8xsi64>> : !mo.tensor<[2, 2, 2, 1, 1], si64, {layout = #mo.layout<QRSFC>}>
  %st = mo.constant <#M.dense_array<1, 1 ,1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %di = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>> : !mo.tensor<[6], si64>
  %op = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 2, 1], si64>, !mo.tensor<[2, 2, 2, 1, 1], si64, {layout = #mo.layout<QRSFC>}>,
    !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[1, 3, 3, 3, 1], si64>
  mo.output %output : !mo.tensor<[1, 3, 3, 3, 1], si64>
}
// CHECK{LITERAL}: 'conv3d_transpose_const_filter' returned tensor<1x3x3x3x1xsi64> [0, 1, 1, 2, 6, 4, 2, 5, 3, 4, 10, 6, 12, 28, 16, 8, 18, 10, 4, 9, 5, 10, 22, 12, 6, 13, 7]

mo.graph @conv3d_transpose_stride(%input: !mo.tensor<[1, 2, 2, 2, 1],si64>, %filter: !mo.tensor<[3, 2, 1, 1, 1],si64, {layout = #mo.layout<QRSFC>}>) -> (!mo.tensor<[1, 4, 4, 4, 1], si64>) {
  %st = mo.constant <#M.dense_array<1, 2 ,3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %di = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>> : !mo.tensor<[6], si64>
  %op = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 2, 1], si64>, !mo.tensor<[3, 2, 1, 1, 1], si64, {layout = #mo.layout<QRSFC>}>,
    !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[1, 4, 4, 4, 1], si64>
  mo.output %output : !mo.tensor<[1, 4, 4, 4, 1], si64>
}
// CHECK{LITERAL}: 'conv3d_transpose_stride' returned tensor<1x4x4x4x1xsi64> [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

mo.graph @conv3d_transpose_pad(%input: !mo.tensor<[1, 2, 2, 2, 1],si64>, %filter: !mo.tensor<[2, 2, 2, 1, 1],si64, {layout = #mo.layout<QRSFC>}>) -> (!mo.tensor<[1, 1, 3, 4, 1], si64>) {
  %st = mo.constant <#M.dense_array<1, 2 ,3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %di = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %pa = mo.constant <#M.dense_array<1, 1, 0, 1, 0, 1> : tensor<6xsi64>> : !mo.tensor<[6], si64>
  %op = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %output = mo.conv_transpose(%input, %filter)
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 2, 1], si64>, !mo.tensor<[2, 2, 2, 1, 1], si64, {layout = #mo.layout<QRSFC>}>,
    !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[1, 1, 3, 4, 1], si64>
  mo.output %output : !mo.tensor<[1, 1, 3, 4, 1], si64>
}
// CHECK{LITERAL}: 'conv3d_transpose_pad' returned tensor<1x1x3x4x1xsi64> [2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2]

mo.graph @conv3d_transpose_fallback_padded_shape_func<() -> D0, D1, D2, D3, D4>(
  %input: !mo.tensor<[2, 5, 10, 7, 9], si32>,
  %filter: !mo.tensor<[3, 2, 2, 3, 9], si32, {layout = #mo.layout<QRSFC>}>
) -> !mo.tensor<[D0, D1, D2, D3, D4], si32> {

  %st = mo.constant <#M.dense_array<3, 2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %di = mo.constant <#M.dense_array<2, 3, 2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %pa = mo.constant <#M.dense_array<1, 1, 0, 0, 1, 1> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %op = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>

  mo.invoke_shape_func["mo.conv_transpose"]<() -> D5, D6, D7, D8, D9>(%input, %filter, %st, %di, %pa, %op)
    [dataDeptTensors = [2 : i32, 3 : i32, 4: i32, 5: i32]] :
    (!mo.tensor<[2, 5, 10, 7, 9], si32>, !mo.tensor<[3, 2, 2, 3, 9], si32, {layout = #mo.layout<QRSFC>}>,
     !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[6], si32>, !mo.tensor<[3], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D5, D6, D7, D8, D9], si32>

  mo.output<D5, D6, D7, D8, D9> %val : !mo.tensor<[D5, D6, D7, D8, D9], si32>
}
// CHECK{LITERAL}: 'conv3d_transpose_fallback_padded_shape_func' returned tensor<2x16x24x20x3xsi32>
//
// One way to verify via PyTorch (it's not exactly the same since PT doesn't
// differentiate applies the same # of 0 paddings on both end of an axis):
//   >>> import torch
//   >>> from torch import nn
//   >>> m = nn.ConvTranspose2d(9, 3, (2, 2), stride=(2, 3), dilation=(3, 2), padding=(1, 1), output_padding=(1, 2))
//   >>> m(input).size()
//   torch.Size([32, 3, 21, 60])
