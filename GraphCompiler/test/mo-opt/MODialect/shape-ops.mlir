// RUN: mo-opt %s | FileCheck %s

// CHECK-LABEL: mo.graph @invoke_shape_func_slice
mo.graph @invoke_shape_func_slice<D1, D2 -> D3, D4>(
  %input: !mo.tensor<[D1, D2], f32>,
  %starts: !mo.tensor<[2], si32>,
  %stops: !mo.tensor<[2], si32>,
  %steps: !mo.tensor<[2], si32>
) -> !mo.tensor<[D3, D4], f32> {

  %input_shape = mo.shape_of(%input) : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[2], si32>

  // CHECK: mo.invoke_shape_func["slice_fw_shape_func"]<() -> D5, D6
  mo.invoke_shape_func["slice_fw_shape_func"]<() -> D5, D6>
  (%input_shape, %starts, %stops, %steps)
  : (!mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>)

  // CHECK: mo.slice({{.*}}) : (!mo.tensor<[D1, D2], f32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[D5, D6], f32>
  %out = mo.slice(%input, %starts, %stops, %steps) : (
    !mo.tensor<[D1, D2], f32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[D5, D6], f32>

  mo.output<D5, D6> %out : !mo.tensor<[D5, D6], f32>
}

// CHECK-LABEL: mo.graph @invoke_shape_func_matmul_packing
mo.graph @invoke_shape_func_matmul_packing<D1, D2 -> D3, D4, D5>(
  %input: !mo.tensor<[D1, D2], f32, {layout = #mo.layout<KN>}>
) -> !mo.tensor<[D3, D4, D5], f32, {layout = #mo.layout<KNkni>}> {

  %input_shape = mo.shape_of(%input) : (!mo.tensor<[D1, D2], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2], si32>

  // CHECK: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> D6, D7, D8>
  mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> D6, D7, D8>(%input_shape)
  {transpose = true} : (!mo.tensor<[2], si32>)

  // CHECK: mo.layout.transform({{.*}}) :
  // CHECK-SAME:  !mo.tensor<[D1, D2], f32, {layout = #mo.layout<KN>}> ->
  // CHECK-SAME:  !mo.tensor<[D6, D7, D8], f32, {layout = #mo.layout<KNkni>}>
  %out = mo.layout.transform(%input) :
    !mo.tensor<[D1, D2], f32, {layout = #mo.layout<KN>}> ->
    !mo.tensor<[D6, D7, D8], f32, {layout = #mo.layout<KNkni>}>

  mo.output<D6, D7, D8> %out : !mo.tensor<[D6, D7, D8], f32, {layout = #mo.layout<KNkni>}>
}
