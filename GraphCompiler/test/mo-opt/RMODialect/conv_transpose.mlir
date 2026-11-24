// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: mo.graph @conv_transpose
mo.graph @conv_transpose(%input: !mo.tensor<[1, 2, 2, 1], si64>, %filter: !mo.tensor<[2, 2, 1, 1], si64>) -> (!mo.tensor<[1, 3, 3, 1], si64>) {
  %output = rmo.conv_transpose(%input, %filter) [strides = #mosh<ape[1, 1]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, output_paddings = #mosh<ape[0, 0]> : !mosh.ape] : (!mo.tensor<[1, 2, 2, 1], si64>, !mo.tensor<[2, 2, 1, 1], si64>) -> !mo.tensor<[1, 3, 3, 1], si64>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], si64>
}

// -----

mo.graph @conv_transpose(%input: !mo.tensor<[1, 2, 2, 1], f32>, %filter: !mo.tensor<[2, 2, 1, 1], si64>) -> (!mo.tensor<[1, 3, 3, 1], si64>) {

  // expected-error @below {{Failed to resolve valid dtype: Unsafe cast from si64 to f64. Insert an explicit cast op if this conversion is wanted.}}
  // expected-error @below {{'rmo.conv_transpose' op failed to infer returned types}}
  %output = rmo.conv_transpose(%input, %filter) [strides = #mosh<ape[1, 1]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, output_paddings = #mosh<ape[0, 0]> : !mosh.ape] : (!mo.tensor<[1, 2, 2, 1], f32>, !mo.tensor<[2, 2, 1, 1], si64>) -> !mo.tensor<[1, 3, 3, 1], si64>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], si64>
}

// -----

mo.graph @conv_transpose(%input: !mo.tensor<[1, 2, 2, 1], si64>, %filter: !mo.tensor<[2, 2, 1, 1], si64>) -> (!mo.tensor<[1, 3, 3, 1], si64>) {

  // expected-error @below {{[conv_transpose] requires (len(paddings) == 2 * (input rank - 2))}}
  // expected-error @below {{'rmo.conv_transpose' op failed to infer returned types}}
  %output = rmo.conv_transpose(%input, %filter) [strides = #mosh<ape[1, 1]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0]> : !mosh.ape, output_paddings = #mosh<ape[0, 0]> : !mosh.ape] : (!mo.tensor<[1, 2, 2, 1], si64>, !mo.tensor<[2, 2, 1, 1], si64>) -> !mo.tensor<[1, 3, 3, 1], si64>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], si64>
}
