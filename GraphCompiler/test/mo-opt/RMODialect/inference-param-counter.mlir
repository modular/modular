// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt | mo-opt | FileCheck %s

// Result type inference names the parameters it declares for dim expressions
// with a builder-local counter, so it must not advance the graph's counter.
// None of these graphs declares a parameter, so none prints a counter; each
// is checked up to its closing brace, where a nonzero counter would appear.
// Each graph uses symbolic dims so that inference builds a non-trivial
// expression.

// CHECK-LABEL: mo.graph @concat_expr<
// CHECK: {{^  }$}}
mo.graph @concat_expr<a, b>(%x: !mo.tensor<[a, 4], f32>, %y: !mo.tensor<[b, 4], f32>) -> !mo.tensor<[add(a, b), 8], f32> {
  %z = rmo.concat(%x, %y) [axis = 0] : (!mo.tensor<[a, 4], f32>, !mo.tensor<[b, 4], f32>) -> !mo.tensor<[add(a, b), 4], f32>
  %w = rmo.concat(%y, %x) [axis = 0] : (!mo.tensor<[b, 4], f32>, !mo.tensor<[a, 4], f32>) -> !mo.tensor<[add(b, a), 4], f32>
  %r = rmo.concat(%z, %w) [axis = 1] : (!mo.tensor<[add(a, b), 4], f32>, !mo.tensor<[add(b, a), 4], f32>) -> !mo.tensor<[add(a, b), 8], f32>
  mo.output %r : !mo.tensor<[add(a, b), 8], f32>
}

// CHECK-LABEL: mo.graph @reshape_symbolic<
// CHECK: {{^  }$}}
mo.graph @reshape_symbolic<a, b>(%x: !mo.tensor<[a, b, 6], f32>) -> !mo.tensor<[mul_no_wrap(a, b), 6], f32> {
  %r = rmo.reshape(%x) [newShape = #mosh<ape[mul_no_wrap(a, b), 6]> : !mosh.ape] : (!mo.tensor<[a, b, 6], f32>) -> !mo.tensor<[mul_no_wrap(a, b), 6], f32>
  mo.output %r : !mo.tensor<[mul_no_wrap(a, b), 6], f32>
}

// CHECK-LABEL: mo.graph @add_expr<
// CHECK: {{^  }$}}
mo.graph @add_expr<a, b>(%x: !mo.tensor<[a, 4], f32>, %y: !mo.tensor<[b, 4], f32>) -> !mo.tensor<[add(a, b), 4], f32> {
  %z = rmo.concat(%x, %y) [axis = 0] : (!mo.tensor<[a, 4], f32>, !mo.tensor<[b, 4], f32>) -> !mo.tensor<[add(a, b), 4], f32>
  %w = rmo.concat(%y, %x) [axis = 0] : (!mo.tensor<[b, 4], f32>, !mo.tensor<[a, 4], f32>) -> !mo.tensor<[add(b, a), 4], f32>
  %r = rmo.add(%z, %w) : (!mo.tensor<[add(a, b), 4], f32>, !mo.tensor<[add(b, a), 4], f32>) -> !mo.tensor<[add(a, b), 4], f32>
  mo.output %r : !mo.tensor<[add(a, b), 4], f32>
}

// CHECK-LABEL: mo.graph @select_expr<
// CHECK: {{^  }$}}
mo.graph @select_expr<a, b>(%x: !mo.tensor<[a, 4], f32>, %y: !mo.tensor<[b, 4], f32>) -> !mo.tensor<[add(a, b), 4], f32> {
  %z = rmo.concat(%x, %y) [axis = 0] : (!mo.tensor<[a, 4], f32>, !mo.tensor<[b, 4], f32>) -> !mo.tensor<[add(a, b), 4], f32>
  %w = rmo.concat(%y, %x) [axis = 0] : (!mo.tensor<[b, 4], f32>, !mo.tensor<[a, 4], f32>) -> !mo.tensor<[add(b, a), 4], f32>
  %c = rmo.equal(%z, %w) : (!mo.tensor<[add(a, b), 4], f32>, !mo.tensor<[add(b, a), 4], f32>) -> !mo.tensor<[add(a, b), 4], bool>
  %r = rmo.select(%c, %z, %w) : (!mo.tensor<[add(a, b), 4], bool>, !mo.tensor<[add(a, b), 4], f32>, !mo.tensor<[add(b, a), 4], f32>) -> !mo.tensor<[add(a, b), 4], f32>
  mo.output %r : !mo.tensor<[add(a, b), 4], f32>
}

// CHECK-LABEL: mo.graph @matmul_expr<
// CHECK: {{^  }$}}
mo.graph @matmul_expr<a, b>(%x: !mo.tensor<[a, 4, 4], f32>, %y: !mo.tensor<[b, 4, 4], f32>) -> !mo.tensor<[add(a, b), 4, 4], f32> {
  %z = rmo.concat(%x, %y) [axis = 0] : (!mo.tensor<[a, 4, 4], f32>, !mo.tensor<[b, 4, 4], f32>) -> !mo.tensor<[add(a, b), 4, 4], f32>
  %w = rmo.concat(%y, %x) [axis = 0] : (!mo.tensor<[b, 4, 4], f32>, !mo.tensor<[a, 4, 4], f32>) -> !mo.tensor<[add(b, a), 4, 4], f32>
  %r = rmo.matmul(%z, %w) : (!mo.tensor<[add(a, b), 4, 4], f32>, !mo.tensor<[add(b, a), 4, 4], f32>) -> !mo.tensor<[add(a, b), 4, 4], f32>
  mo.output %r : !mo.tensor<[add(a, b), 4, 4], f32>
}

// CHECK-LABEL: mo.graph @slice_symbolic<
// CHECK: {{^  }$}}
mo.graph @slice_symbolic<a>(%x: !mo.tensor<[a], f32>) -> !mo.tensor<[add(a, -1)], f32> {
  %r = rmo.slice(%x) [starts = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[a]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape] : (!mo.tensor<[a], f32>) -> !mo.tensor<[add(a, -1)], f32>
  mo.output %r : !mo.tensor<[add(a, -1)], f32>
}

// CHECK-LABEL: mo.graph @broadcast_to_symbolic<
// CHECK: {{^  }$}}
mo.graph @broadcast_to_symbolic<a>(%x: !mo.tensor<[1, 4], f32>) -> !mo.tensor<[a, 4], f32> {
  %r = rmo.broadcast_to(%x) [newShape = #mosh<ape[a, 4]> : !mosh.ape] : (!mo.tensor<[1, 4], f32>) -> !mo.tensor<[a, 4], f32>
  mo.output %r : !mo.tensor<[a, 4], f32>
}

// CHECK-LABEL: mo.graph @conv_symbolic<
// CHECK: {{^  }$}}
mo.graph @conv_symbolic<n, h, w>(%input: !mo.tensor<[n, h, w, 32], f32>, %filter: !mo.tensor<[2, 2, 32, 64], f32>) -> !mo.tensor<[n, add(h, -1), add(w, -1), 64], f32> {
  %out = rmo.conv(%input, %filter) [strides = #mosh<ape[1, 1]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, num_groups = 1] : (!mo.tensor<[n, h, w, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>) -> !mo.tensor<[n, add(h, -1), add(w, -1), 64], f32>
  mo.output %out : !mo.tensor<[n, add(h, -1), add(w, -1), 64], f32>
}

// CHECK-LABEL: mo.graph @conv_transpose_symbolic<
// CHECK: {{^  }$}}
mo.graph @conv_transpose_symbolic<n, h, w>(%input: !mo.tensor<[n, h, w, 1], f32>, %filter: !mo.tensor<[2, 2, 1, 1], f32>) -> !mo.tensor<[n, add(h, 1), add(w, 1), 1], f32> {
  %out = rmo.conv_transpose(%input, %filter) [strides = #mosh<ape[1, 1]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, output_paddings = #mosh<ape[0, 0]> : !mosh.ape] : (!mo.tensor<[n, h, w, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32>) -> !mo.tensor<[n, add(h, 1), add(w, 1), 1], f32>
  mo.output %out : !mo.tensor<[n, add(h, 1), add(w, 1), 1], f32>
}

// CHECK-LABEL: mo.graph @max_pool_symbolic<
// CHECK: {{^  }$}}
mo.graph @max_pool_symbolic<n, h>(%input: !mo.tensor<[n, h, 15, 1], f32>) -> !mo.tensor<[n, div(h, 3), 7, 1], f32> {
  %out = rmo.max_pool(%input) [filter_shape = #mosh<ape[3, 2]> : !mosh.ape, strides = #mosh<ape[3, 2]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, ceil_mode = false]: (!mo.tensor<[n, h, 15, 1], f32>) -> !mo.tensor<[n, div(h, 3), 7, 1], f32>
  mo.output %out : !mo.tensor<[n, div(h, 3), 7, 1], f32>
}

// CHECK-LABEL: mo.graph @avg_pool_symbolic<
// CHECK: {{^  }$}}
mo.graph @avg_pool_symbolic<n, h>(%input: !mo.tensor<[n, h, 15, 1], f32>) -> !mo.tensor<[n, div(h, 3), 7, 1], f32> {
  %out = rmo.avg_pool(%input) [filter_shape = #mosh<ape[3, 2]> : !mosh.ape, strides = #mosh<ape[3, 2]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, ceil_mode = false, count_boundary = true]: (!mo.tensor<[n, h, 15, 1], f32>) -> !mo.tensor<[n, div(h, 3), 7, 1], f32>
  mo.output %out : !mo.tensor<[n, div(h, 3), 7, 1], f32>
}
