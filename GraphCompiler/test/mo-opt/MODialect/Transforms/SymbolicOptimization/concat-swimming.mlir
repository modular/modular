// RUN: mo-opt --mo-symbolic-optimize %s -split-input-file | FileCheck %s

// CHECK-LABEL:  mo.graph @reshape_last_dim
// CHECK-NEXT:     %[[RESHAPE1:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
// CHECK-NEXT:     %[[RESHAPE2:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[2] (%[[RESHAPE1]], %[[RESHAPE2]]) : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[2, 2, 4], f32>) -> !mo.tensor<[2, 2, 8], f32>
// CHECK-NEXT:     mo.output %[[CONCAT]] : !mo.tensor<[2, 2, 8], f32>
// CHECK-NEXT:  }

mo.graph @reshape_last_dim(%in: !mo.tensor<[2, 8], f32>) -> (!mo.tensor<[2, 2, 8], f32>) {
  %1 = mo.static.reshape(%in) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
  %3 = mo.concat[2] (%1, %2) : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[2, 2, 4], f32>) -> (!mo.tensor<[2, 2, 8], f32>)
  mo.output %3 : !mo.tensor<[2, 2, 8], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_not_last_axis
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[1] ({{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 16], f32>
// CHECK-NEXT:     %[[RESHAPE:.*]] = mo.static.reshape(%[[CONCAT]]) : !mo.tensor<[2, 16], f32> -> !mo.tensor<[2, 4, 4], f32>
// CHECK-NEXT:     mo.output %[[RESHAPE]] : !mo.tensor<[2, 4, 4], f32>

mo.graph @reshape_not_last_axis(%in: !mo.tensor<[2, 8], f32>) -> (!mo.tensor<[2, 4, 4], f32>) {
  %1 = mo.static.reshape(%in) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
  %3 = mo.concat[1] (%1, %2) : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[2, 2, 4], f32>) -> (!mo.tensor<[2, 4, 4], f32>)
  mo.output %3 : !mo.tensor<[2, 4, 4], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_1_in_middle
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[1] ({{.*}}) : (!mo.tensor<[2, 2, 1, 1, 2, 2], f32>, !mo.tensor<[2, 2, 1, 1, 2, 2], f32>) -> !mo.tensor<[2, 4, 1, 1, 2, 2], f32>
// CHECK-NEXT:     %[[RESHAPE:.*]] = mo.static.reshape(%[[CONCAT]]) : !mo.tensor<[2, 4, 1, 1, 2, 2], f32> -> !mo.tensor<[2, 8, 2], f32>
// CHECK-NEXT:     mo.output %[[RESHAPE]] : !mo.tensor<[2, 8, 2], f32>

mo.graph @reshape_1_in_middle(%in: !mo.tensor<[2, 2, 1, 1, 2, 2], f32>) -> (!mo.tensor<[2, 8, 2], f32>) {
  %1 = mo.static.reshape(%in) : !mo.tensor<[2, 2, 1, 1, 2, 2], f32> -> !mo.tensor<[2, 4, 2], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[2, 2, 1, 1, 2, 2], f32> -> !mo.tensor<[2, 4, 2], f32>
  %3 = mo.concat[1] (%1, %2) : (!mo.tensor<[2, 4, 2], f32>, !mo.tensor<[2, 4, 2], f32>) -> (!mo.tensor<[2, 8, 2], f32>)
  mo.output %3 : !mo.tensor<[2, 8, 2], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_mixed_but_look_right
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[0] ({{.*}}) : (!mo.tensor<[100], f32>, !mo.tensor<[100], f32>) -> !mo.tensor<[200], f32>
// CHECK-NEXT:     %[[RESHAPE:.*]] = mo.static.reshape(%[[CONCAT]]) : !mo.tensor<[200], f32> -> !mo.tensor<[2, 100], f32>
// CHECK-NEXT:     mo.output %[[RESHAPE]] : !mo.tensor<[2, 100], f32>

mo.graph @reshape_mixed_but_look_right(%in: !mo.tensor<[100], f32>) -> (!mo.tensor<[2, 100], f32>) {
  %1 = mo.static.reshape(%in) : !mo.tensor<[100], f32> -> !mo.tensor<[1, 100], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[100], f32> -> !mo.tensor<[1, 100], f32>
  %3 = mo.concat[1] (%1, %2) : (!mo.tensor<[1, 100], f32>, !mo.tensor<[1, 100], f32>) -> (!mo.tensor<[2, 100], f32>)
  mo.output %3 : !mo.tensor<[2, 100], f32>
}

// -----

// CHECK-LABEL: @reshape_no_sinking_1_suffix_mixed_dimensions
// CHECK-NEXT:     %[[RESHAPE1:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[100], f32> -> !mo.tensor<[100, 1], f32>
// CHECK-NEXT:     %[[RESHAPE2:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[100], f32> -> !mo.tensor<[100, 1], f32>
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[1] (%[[RESHAPE1]], %[[RESHAPE2]]) : (!mo.tensor<[100, 1], f32>, !mo.tensor<[100, 1], f32>) -> !mo.tensor<[100, 2], f32>
// CHECK-NEXT:     mo.output %[[CONCAT]] : !mo.tensor<[100, 2], f32>

mo.graph @reshape_no_sinking_1_suffix_mixed_dimensions(%in: !mo.tensor<[100], f32>) -> (!mo.tensor<[100, 2], f32>) {
  %1 = mo.static.reshape(%in) : !mo.tensor<[100], f32> -> !mo.tensor<[100, 1], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[100], f32> -> !mo.tensor<[100, 1], f32>
  %3 = mo.concat[1] (%1, %2) : (!mo.tensor<[100, 1], f32>, !mo.tensor<[100, 1], f32>) -> (!mo.tensor<[100, 2], f32>)
  mo.output %3 : !mo.tensor<[100, 2], f32>
}

// -----

// CHECK-LABEL:  mo.graph @reshape_no_sinking_different_shapes
// CHECK-NEXT:     %[[RESHAPE1:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[50, 2], f32> -> !mo.tensor<[100, 1], f32>
// CHECK-NEXT:     %[[RESHAPE2:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[100], f32> -> !mo.tensor<[100, 1], f32>
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[1] (%[[RESHAPE1]], %[[RESHAPE2]]) : (!mo.tensor<[100, 1], f32>, !mo.tensor<[100, 1], f32>) -> !mo.tensor<[100, 2], f32>
// CHECK-NEXT:     mo.output %[[CONCAT]] : !mo.tensor<[100, 2], f32>

mo.graph @reshape_no_sinking_different_shapes(%in1: !mo.tensor<[50, 2], f32>, %in2: !mo.tensor<[100], f32>) -> (!mo.tensor<[100, 2], f32>) {
  %1 = mo.static.reshape(%in1) : !mo.tensor<[50, 2], f32> -> !mo.tensor<[100, 1], f32>
  %2 = mo.static.reshape(%in2) : !mo.tensor<[100], f32> -> !mo.tensor<[100, 1], f32>
  %3 = mo.concat[1] (%1, %2) : (!mo.tensor<[100, 1], f32>, !mo.tensor<[100, 1], f32>) -> (!mo.tensor<[100, 2], f32>)
  mo.output %3 : !mo.tensor<[100, 2], f32>
}

// -----

// CHECK-LABEL: mo.graph @reshapes_used_elsewhere
// CHECK-NEXT:     %[[RESHAPE1:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[64], f32> -> !mo.tensor<[1, 64, 1], f32>
// CHECK-NEXT:     %[[RESHAPE2:.*]] = mo.static.reshape({{.*}}) : !mo.tensor<[64], f32> -> !mo.tensor<[1, 64, 1], f32>
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[0] (%[[RESHAPE1]], %[[RESHAPE2]]) : (!mo.tensor<[1, 64, 1], f32>, !mo.tensor<[1, 64, 1], f32>) -> !mo.tensor<[2, 64, 1], f32>
// CHECK-NEXT:     mo.output %[[CONCAT]], %[[RESHAPE1]] : !mo.tensor<[2, 64, 1], f32>, !mo.tensor<[1, 64, 1], f32>

mo.graph @reshapes_used_elsewhere(%in: !mo.tensor<[64], f32>) -> (!mo.tensor<[2, 64, 1], f32>, !mo.tensor<[1, 64, 1], f32>) {
  %1 = mo.static.reshape(%in) : !mo.tensor<[64], f32> -> !mo.tensor<[1, 64, 1], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[64], f32> -> !mo.tensor<[1, 64, 1], f32>
  %3 = mo.concat[0] (%1, %2) : (!mo.tensor<[1, 64, 1], f32>, !mo.tensor<[1, 64, 1], f32>) -> (!mo.tensor<[2, 64, 1], f32>)
  mo.output %3, %1 : !mo.tensor<[2, 64, 1], f32>, !mo.tensor<[1, 64, 1], f32>
}

// -----

// CHECK-LABEL: mo.graph @rebind
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[2] ({{.*}}) : (!mo.tensor<[D1, 2, 2], f32>, !mo.tensor<[D1, 2, 2], f32>) -> !mo.tensor<[D1, 2, 4], f32>
// CHECK-NEXT:     %[[REBIND:.*]] = mo.rebind(%[[CONCAT]]) : !mo.tensor<[D1, 2, 4], f32> -> !mo.tensor<[D0, 2, 4], f32>
// CHECK-NEXT:     mo.output %[[REBIND]] : !mo.tensor<[D0, 2, 4], f32>

mo.graph @rebind<D0, D1>(%in1: !mo.tensor<[D1, 2, 2], f32>, %in2: !mo.tensor<[D1, 2, 2], f32>) -> (!mo.tensor<[D0, 2, 4], f32>) {
  %1 = mo.rebind(%in1) : !mo.tensor<[D1, 2, 2], f32> -> !mo.tensor<[D0, 2, 2], f32>
  %2 = mo.rebind(%in2) : !mo.tensor<[D1, 2, 2], f32> -> !mo.tensor<[D0, 2, 2], f32>
  %3 = mo.concat[2] (%1, %2) : (!mo.tensor<[D0, 2, 2], f32>, !mo.tensor<[D0, 2, 2], f32>) -> (!mo.tensor<[D0, 2, 4], f32>)
  mo.output %3 : !mo.tensor<[D0, 2, 4], f32>
}

// -----

// CHECK-LABEL: mo.graph @F2_like
// CHECK-NEXT:     %[[CONCAT:.*]] = mo.concat[0] ({{.*}}) : (!mo.tensor<[64], f32>, !mo.tensor<[64], f32>) -> !mo.tensor<[128], f32>
// CHECK-NEXT:     %[[RESHAPE:.*]] = mo.static.reshape(%[[CONCAT]]) : !mo.tensor<[128], f32> -> !mo.tensor<[2, 64, 1], f32>
// CHECK-NEXT:     mo.output %[[RESHAPE]] : !mo.tensor<[2, 64, 1], f32>

mo.graph @F2_like(%in: !mo.tensor<[64], f32>) -> (!mo.tensor<[2, 64, 1], f32>) {
  %1 = mo.static.reshape(%in) : !mo.tensor<[64], f32> -> !mo.tensor<[1, 64, 1], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[64], f32> -> !mo.tensor<[1, 64, 1], f32>
  %3 = mo.concat[0] (%1, %2) : (!mo.tensor<[1, 64, 1], f32>, !mo.tensor<[1, 64, 1], f32>) -> (!mo.tensor<[2, 64, 1], f32>)
  mo.output %3 : !mo.tensor<[2, 64, 1], f32>
}
