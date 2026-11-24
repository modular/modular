// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @conv(%input : !mo.tensor<[1, 5, 5, 3], f32>, %filter : !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[1, 3, 3, 1], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>
  %output = mo.conv(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 3], f32>, !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'conv' returned tensor<1x3x3x1xf32> [5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01]

mo.graph @strided_conv(%input : !mo.tensor<[1, 5, 5, 3], f32>, %filter : !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[1, 2, 2, 1], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<2, 2> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>
  %output = mo.conv(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 3], f32>, !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 2, 2, 1], f32>
  mo.output %output : !mo.tensor<[1, 2, 2, 1], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'strided_conv' returned tensor<1x2x2x1xf32> [5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01]

mo.graph @padded_conv(%input : !mo.tensor<[1, 5, 5, 3], f32>, %filter : !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[1, 5, 3, 1], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<1, 1, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>
  %output = mo.conv(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 3], f32>, !mo.tensor<[3, 3, 3, 1], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 5, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 5, 3, 1], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'padded_conv' returned tensor<1x5x3x1xf32> [3.600000e+01, 3.600000e+01, 3.600000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 5.400000e+01, 3.600000e+01, 3.600000e+01, 3.600000e+01]

mo.graph @grouped_conv(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 2, 2, 3], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
  %ng = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %filter = "mo.constant"() {value = #M.dense_array<1.0, 1.0, 1.0> : tensor<1x1x1x3xf32>} : () -> !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>
  %output = mo.conv(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 2, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 2, 2, 3], f32>
  mo.output %output : !mo.tensor<[1, 2, 2, 3], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'grouped_conv' returned tensor<1x2x2x3xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]

mo.graph @conv3d(%input : !mo.tensor<[1, 5, 5, 5, 2], f32>, %filter : !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>) -> (!mo.tensor<[1, 3, 3, 3, 1], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>} : () -> !mo.tensor<[6], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>
  %output = mo.conv(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 5, 2], f32>, !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 3, 3, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 3, 1], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'conv3d' returned tensor<1x3x3x3x1xf32> [1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02]

mo.graph @strided_conv3d(%input : !mo.tensor<[1, 5, 5, 5, 2], f32>, %filter : !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>) -> (!mo.tensor<[1, 2, 2, 2, 1], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<2, 2, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi64>} : () -> !mo.tensor<[6], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>
  %output = mo.conv(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 5, 2], f32>, !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 2, 2, 2, 1], f32>
  mo.output %output : !mo.tensor<[1, 2, 2, 2, 1], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'strided_conv3d' returned tensor<1x2x2x2x1xf32> [1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02, 1.080000e+02]

mo.graph @padded_conv3d(%input : !mo.tensor<[1, 3, 3, 3, 2], f32>, %filter : !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>) -> (!mo.tensor<[1, 3, 3, 3, 1], f32>) {
  %st = "mo.constant"() {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %di = "mo.constant"() {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %pa = "mo.constant"() {value = #M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>} : () -> !mo.tensor<[6], si64>
  %ng = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () -> !mo.tensor<[], si64>
  %0 = mo.add(%filter, %filter) : !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>
  %output = mo.conv(%input, %0) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 3, 3, 3, 2], f32>, !mo.tensor<[3, 3, 3, 2, 1], f32, {layout = #mo.layout<QRSCF>}>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 3, 3, 3, 1], f32>
  mo.output %output : !mo.tensor<[1, 3, 3, 3, 1], f32>
}
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.conv{{.*}}"
// CHECK{LITERAL}: 'padded_conv3d' returned tensor<1x3x3x3x1xf32> [3.200000e+01, 4.800000e+01, 3.200000e+01, 4.800000e+01, 7.200000e+01, 4.800000e+01, 3.200000e+01, 4.800000e+01, 3.200000e+01, 4.800000e+01, 7.200000e+01, 4.800000e+01, 7.200000e+01, 1.080000e+02, 7.200000e+01, 4.800000e+01, 7.200000e+01, 4.800000e+01, 3.200000e+01, 4.800000e+01, 3.200000e+01, 4.800000e+01, 7.200000e+01, 4.800000e+01, 3.200000e+01, 4.800000e+01, 3.200000e+01]

mo.graph @conv_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter: !mo.tensor<[2, 2, 3, 1], f32, {layout = #mo.layout<RSCF>}>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %strides = mo.constant <#M.dense_array<2, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %dilations = mo.constant <#M.dense_array<3, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  mo.invoke_shape_func["mo.conv"]<() -> D4, D5, D6, D7>(%input, %filter, %strides, %dilations, %paddings, %num_groups)
    [dataDeptTensors = [2 : i32, 3 : i32, 4: i32, 5: i32]] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32, {layout = #mo.layout<RSCF>}>,
     !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[4], si32>, !mo.tensor<[], si64>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'conv_fallback_shape_func' returned tensor<32x4x6x1xsi32>

mo.graph @conv_fallback_padded_shape_func<() -> D0, D1, D2, D3>(
  %input: !mo.tensor<[32, 10, 20, 3], f32>,
  %filter: !mo.tensor<[2, 2, 3, 1], f32, {layout = #mo.layout<RSCF>}>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %strides = mo.constant <#M.dense_array<2, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %dilations = mo.constant <#M.dense_array<3, 2> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %paddings = mo.constant <#M.dense_array<1, 1, 0, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  mo.invoke_shape_func["mo.conv"]<() -> D4, D5, D6, D7>(%input, %filter, %strides, %dilations, %paddings, %num_groups)
    [dataDeptTensors = [2 : i32, 3 : i32, 4: i32, 5: i32]] :
    (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2, 2, 3, 1], f32, {layout = #mo.layout<RSCF>}>,
     !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[4], si32>, !mo.tensor<[], si64>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
// CHECK{LITERAL}: 'conv_fallback_padded_shape_func' returned tensor<32x5x6x1xsi32>

mo.graph @conv3d_fallback_shape_func<() -> D0, D1, D2, D3, D4>(
  %input: !mo.tensor<[2, 10, 20, 7, 3], f32>,
  %filter: !mo.tensor<[2, 2, 3, 3, 1], f32, {layout = #mo.layout<QRSCF>}>
) -> !mo.tensor<[D0, D1, D2, D3, D4], si32> {

  %strides = mo.constant <#M.dense_array<2, 3, 2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %dilations = mo.constant <#M.dense_array<3, 2, 2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  mo.invoke_shape_func["mo.conv"]<() -> D5, D6, D7, D8, D9>(%input, %filter, %strides, %dilations, %paddings, %num_groups)
    [dataDeptTensors = [2 : i32, 3 : i32, 4: i32, 5: i32]] :
    (!mo.tensor<[2, 10, 20, 7, 3], f32>, !mo.tensor<[2, 2, 3, 3, 1], f32, {layout = #mo.layout<QRSCF>}>,
     !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[6], si32>, !mo.tensor<[], si64>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D5, D6, D7, D8, D9], si32>

  mo.output<D5, D6, D7, D8, D9> %val : !mo.tensor<[D5, D6, D7, D8, D9], si32>
}
// CHECK{LITERAL}: 'conv3d_fallback_shape_func' returned tensor<2x4x6x2x1xsi32>

mo.graph @conv3d_fallback_padded_shape_func<() -> D0, D1, D2, D3, D4>(
  %input: !mo.tensor<[2, 10, 20, 7, 3], f32>,
  %filter: !mo.tensor<[2, 2, 3, 3, 1], f32, {layout = #mo.layout<QRSCF>}>
) -> !mo.tensor<[D0, D1, D2, D3, D4], si32> {

  %strides = mo.constant <#M.dense_array<2, 3, 2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %dilations = mo.constant <#M.dense_array<3, 2, 2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %paddings = mo.constant <#M.dense_array<3, 0, 0, 0, 2, 2> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  mo.invoke_shape_func["mo.conv"]<() -> D5, D6, D7, D8, D9>(%input, %filter, %strides, %dilations, %paddings, %num_groups)
    [dataDeptTensors = [2 : i32, 3 : i32, 4: i32, 5: i32]] :
    (!mo.tensor<[2, 10, 20, 7, 3], f32>, !mo.tensor<[2, 2, 3, 3, 1], f32, {layout = #mo.layout<QRSCF>}>,
     !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[6], si32>, !mo.tensor<[], si64>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D5, D6, D7, D8, D9], si32>

  mo.output<D5, D6, D7, D8, D9> %val : !mo.tensor<[D5, D6, D7, D8, D9], si32>
}
// CHECK{LITERAL}: 'conv3d_fallback_padded_shape_func' returned tensor<2x5x6x4x1xsi32>
