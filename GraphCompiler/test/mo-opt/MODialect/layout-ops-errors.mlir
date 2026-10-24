// RUN: mo-opt %s --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// LayoutTransformOp
//===----------------------------------------------------------------------===//

mo.graph @layout_transform_missing_input_layout(
  %input: !mo.tensor<[?, ?, ?, ?], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}> {

  // expected-error @below {{input type must contain layout annotation}}
  %res = mo.layout.transform(%input) :
    !mo.tensor<[?, ?, ?, ?], f32> ->
    !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>

  mo.output %res : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>
}

// -----

mo.graph @layout_transform_missing_output_layout(
  %input: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  // expected-error @below {{output type must contain layout annotation}}
  %res = mo.layout.transform(%input) :
    !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}> ->
    !mo.tensor<[?, ?, ?, ?], f32>

  mo.output %res : !mo.tensor<[?, ?, ?, ?], f32>
}
