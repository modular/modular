// RUN: mo-opt %s --mo-infer-layouts --verify-diagnostics

mo.graph @conv_same_input_and_filter(
  %input: !mo.tensor<[?, ?, ?, ?], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>,
  %num_groups: !mo.tensor<[], si64>
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  // expected-error @below {{attempting to constrain value to layout: #mo.layout<FCRS> when it's already constrained to layout: #mo.layout<NHWC>}}
  %conv1 = mo.conv(%input, %input) [strides = %strides, dilations = %dilations, paddings = %paddings, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  mo.output %conv1 : !mo.tensor<[?, ?, ?, ?], f32>
}
