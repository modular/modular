// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: mo.graph @conv_simple
mo.graph @conv_simple(
  %input: !mo.tensor<[10, 5, 5, 32], f32>,
  %filter: !mo.tensor<[2, 2, 32, 64], f32>
) -> !mo.tensor<[10, 4, 4, 64], f32> {


  // CHECK: %[[OUT:.*]] = rmo.conv
  %out = rmo.conv(%input, %filter) [strides = #mosh<ape[1, 1]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, num_groups = 1] : (!mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>) -> !mo.tensor<[10, 4, 4, 64], f32>
  mo.output %out : !mo.tensor<[10, 4, 4, 64], f32>
}

// -----

mo.graph @conv_error(
  %input: !mo.tensor<[10, 5, 5, 32], f32>,
  %filter: !mo.tensor<[2, 2, 32, 64], f32>
) -> !mo.tensor<[10, 4, 4, 64], f32> {

  // expected-error @below {{'rmo.conv' op inferred type(s) '!mo.tensor<[10, 2, 2, 64], f32>' are incompatible with return type(s) of operation '!mo.tensor<[10, 4, 4, 64], f32>'}}
  // expected-error @below {{'rmo.conv' op failed to infer returned types}}
  %out = rmo.conv(%input, %filter) [strides = #mosh<ape[2, 2]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, num_groups = 1] : (!mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>) -> !mo.tensor<[10, 4, 4, 64], f32>
  mo.output %out : !mo.tensor<[10, 4, 4, 64], f32>
}
