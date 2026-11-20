// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: mo.graph @max_pool
mo.graph @max_pool(
  %input: !mo.tensor<[1, 6, 15, 1], f32>
) -> !mo.tensor<[1, 2, 7, 1], f32> {


  // CHECK: %[[OUT:.*]] = rmo.max_pool
  %out = rmo.max_pool(%input) [filter_shape = #mosh<ape[3, 2]> : !mosh.ape, strides = #mosh<ape[3, 2]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, ceil_mode = false]: (!mo.tensor<[1, 6, 15, 1], f32>) -> !mo.tensor<[1, 2, 7, 1], f32>
  mo.output %out : !mo.tensor<[1, 2, 7, 1], f32>
}

// -----

mo.graph @max_pool_error(
  %input: !mo.tensor<[1, 6, 15, 1], f32>
) -> !mo.tensor<[1, 2, 7, 1], f32> {


  // expected-error @below {{'rmo.max_pool' op inferred type(s) '!mo.tensor<[1, 2, 8, 1], f32>' are incompatible with return type(s) of operation '!mo.tensor<[1, 2, 7, 1], f32>'}}
  // expected-error @below {{'rmo.max_pool' op failed to infer returned types}}
  %out = rmo.max_pool(%input) [filter_shape = #mosh<ape[3, 2]> : !mosh.ape, strides = #mosh<ape[3, 2]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, ceil_mode = true]: (!mo.tensor<[1, 6, 15, 1], f32>) -> !mo.tensor<[1, 2, 7, 1], f32>
  mo.output %out : !mo.tensor<[1, 2, 7, 1], f32>
}

// -----

// CHECK-LABEL: mo.graph @avg_pool
mo.graph @avg_pool(
  %input: !mo.tensor<[1, 6, 15, 1], f32>
) -> !mo.tensor<[1, 2, 7, 1], f32> {


  // CHECK: %[[OUT:.*]] = rmo.avg_pool
  %out = rmo.avg_pool(%input) [filter_shape = #mosh<ape[3, 2]> : !mosh.ape, strides = #mosh<ape[3, 2]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, ceil_mode = false, count_boundary = true]: (!mo.tensor<[1, 6, 15, 1], f32>) -> !mo.tensor<[1, 2, 7, 1], f32>
  mo.output %out : !mo.tensor<[1, 2, 7, 1], f32>
}

// -----

mo.graph @avg_pool_error(
  %input: !mo.tensor<[1, 6, 15, 1], f32>
) -> !mo.tensor<[1, 2, 7, 1], f32> {


  // expected-error @below {{'rmo.avg_pool' op inferred type(s) '!mo.tensor<[1, 2, 8, 1], f32>' are incompatible with return type(s) of operation '!mo.tensor<[1, 2, 7, 1], f32>'}}
  // expected-error @below {{'rmo.avg_pool' op failed to infer returned types}}
  %out = rmo.avg_pool(%input) [filter_shape = #mosh<ape[3, 2]> : !mosh.ape, strides = #mosh<ape[3, 2]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, ceil_mode = true, count_boundary = true]: (!mo.tensor<[1, 6, 15, 1], f32>) -> !mo.tensor<[1, 2, 7, 1], f32>
  mo.output %out : !mo.tensor<[1, 2, 7, 1], f32>
}