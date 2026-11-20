// RUN: mo-opt --mo-symbolic-optimize %s | FileCheck %s

// CHECK-LABEL: @symbolic_4d
// CHECK-SAME: %[[ARG0:.*]]: 
// CHECK: %[[PERM:.*]] = mo.constant
// CHECK-SAME: M.dense_array<2, 1, 0, 3>
// CHECK: %[[T0:.*]] = mo.transpose(%[[ARG0]], %[[PERM]])
// CHECK-NEXT: mo.static.reshape(%[[T0]])
mo.graph @symbolic_4d<D1, D2 -> ()>(%arg0: !mo.tensor<[D2, 1, D1, 512], f32>) -> !mo.tensor<[D1, D2, 512], f32>
{
  %perm0 = mo.constant <#M.dense_array<1, 0, 2> : !M.array<3xsi64>> : !mo.tensor<[3], si64>

  %r0 = mo.static.reshape(%arg0) : !mo.tensor<[D2, 1, D1, 512], f32> -> !mo.tensor<[D2, D1, 512], f32>
  %t0 = mo.transpose(%r0, %perm0) : (!mo.tensor<[D2, D1, 512], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D1, D2, 512], f32>

  mo.output %t0 : !mo.tensor<[D1, D2, 512], f32>
}

// -----

// CHECK-LABEL @reshape_not_squeeze_like
// CHECK: mo.constant
// CHECK-NEXT: mo.static.reshape
// CHECK-NEXT: mo.transpose
mo.graph @reshape_not_squeeze_like<D1, D2 -> ()>(%arg0: !mo.tensor<[D2, 2, D1, 512], f32>) -> !mo.tensor<[D1, D2, 1024], f32>
{
  %perm0 = mo.constant <#M.dense_array<1, 0, 2> : !M.array<3xsi64>> : !mo.tensor<[3], si64>

  %r0 = mo.static.reshape(%arg0) : !mo.tensor<[D2, 2, D1, 512], f32> -> !mo.tensor<[D2, D1, 1024], f32>
  %t0 = mo.transpose(%r0, %perm0) : (!mo.tensor<[D2, D1, 1024], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D1, D2, 1024], f32>

  mo.output %t0 : !mo.tensor<[D1, D2, 1024], f32>
}

// -----

!final_axis_input_type = !mo.tensor<[5, 4, 1], f32>
!final_axis_reshaped_type = !mo.tensor<[5, 4], f32>
!final_axis_transposed_type = !mo.tensor<[4, 5], f32>

// CHECK-LABEL: @reshape_squeezed_final_axis
// CHECK-SAME: %[[ARG0:.*]]: 
// CHECK: %[[PERM:.*]] = mo.constant
// CHECK-SAME: M.dense_array<1, 0, 2>
// CHECK: %[[T0:.*]] = mo.transpose(%[[ARG0]], %[[PERM]])
// CHECK-NEXT: mo.static.reshape(%[[T0]])
mo.graph @reshape_squeezed_final_axis(%arg0: !final_axis_input_type) -> !final_axis_transposed_type
{
  %perm0 = mo.constant <#M.dense_array<1, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>

  %r0 = mo.static.reshape(%arg0) : !final_axis_input_type -> !final_axis_reshaped_type
  %t0 = mo.transpose(%r0, %perm0) : (!final_axis_reshaped_type, !mo.tensor<[2], si64>) -> !final_axis_transposed_type

  mo.output %t0 : !final_axis_transposed_type
}

// -----

!first_axis_input_type = !mo.tensor<[1, 5, 4], f32>
!first_axis_reshaped_type = !mo.tensor<[5, 4], f32>
!first_axis_transposed_type = !mo.tensor<[4, 5], f32>

// CHECK-LABEL: @reshape_squeezed_first_axis
// CHECK-SAME: %[[ARG0:.*]]: 
// CHECK: %[[PERM:.*]] = mo.constant
// CHECK-SAME: M.dense_array<0, 2, 1>
// CHECK: %[[T0:.*]] = mo.transpose(%[[ARG0]], %[[PERM]])
// CHECK-NEXT: mo.static.reshape(%[[T0]])
mo.graph @reshape_squeezed_first_axis(%arg0: !first_axis_input_type) -> !first_axis_transposed_type
{
  %perm0 = mo.constant <#M.dense_array<1, 0> : !M.array<2xsi64>> : !mo.tensor<[2], si64>

  %r0 = mo.static.reshape(%arg0) : !first_axis_input_type -> !first_axis_reshaped_type
  %t0 = mo.transpose(%r0, %perm0) : (!first_axis_reshaped_type, !mo.tensor<[2], si64>) -> !first_axis_transposed_type

  mo.output %t0 : !first_axis_transposed_type
}
