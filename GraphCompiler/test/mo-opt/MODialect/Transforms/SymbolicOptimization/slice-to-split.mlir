// RUN: mo-opt -split-input-file --mo-symbolic-optimize %s | FileCheck %s

// CHECK-LABEL:  mo.graph @usual_case
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%{{.*}}, %[[SIZES]]) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
// CHECK-NEXT:  }

mo.graph @usual_case(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Negative bounds count back from the end of the axis, so these two slices are
// the same partition as @usual_case and must keep that order: the slice
// starting at 0 is result #0. Ordering on the raw start constant would put the
// -2 slice first and swap the results (GEX-4199).

// CHECK-LABEL:  mo.graph @negative_bounds
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%{{.*}}, %[[SIZES]]) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>

mo.graph @negative_bounds(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, -2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, -2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_conversion_dynamic_start
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_dynamic_start(%in: !mo.tensor<[2, 4], f32>, %start1: !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start2 = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_conversion_multidimensional_slice
// CHECK:          mo.slice
// CHECK:          mo.slice
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_multidimensional_slice(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>) {
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start3 = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start4 = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<1, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop3 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop4 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2], f32>
  %4 = mo.slice(%in, %start3, %stop3, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2], f32>
  %5 = mo.slice(%in, %start4, %stop4, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[1, 2], f32>
  mo.output %2, %3, %4, %5 : !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_conversion_overlap
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_overlap(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL:  mo.graph @no_conversion_partial_coverage
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_partial_coverage(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 1], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 1], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 1], f32>, !mo.tensor<[2, 2], f32>
}

// An out-of-bound start is clamped onto the axis, so both of these slices
// start at 0 and overlap rather than tile it.

// CHECK-LABEL:  mo.graph @no_conversion_clamped_starts_overlap
// CHECK-NOT:      mo.split
// CHECK:          mo.slice
// CHECK:          mo.slice

mo.graph @no_conversion_clamped_starts_overlap(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant <#M.dense_array<0, -100> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// A start of -100 on a 4-wide axis clamps to 0, the same leading chunk a start
// of 0 would name, so it pairs with the slice starting at 2 to tile the axis.

// CHECK-LABEL:  mo.graph @clamped_negative_start_folds
// CHECK-NEXT:    %[[SIZES:.*]] = mo.constant <#M.dense_array<2, 2> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:    %[[SPLIT:.*]]:2 = mo.split(%{{.*}}, %[[SIZES]]) [axis = 1] : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    mo.output %[[SPLIT]]#0, %[[SPLIT]]#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>

mo.graph @clamped_negative_start_folds(%in: !mo.tensor<[2, 4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start1 = mo.constant <#M.dense_array<0, -100> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start2 = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop2 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.slice(%in, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.slice(%in, %start2, %stop2, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Test that split sizes constant is created on host even with device tensor input
mo.graph @slice_to_split_host_constant<N>(%input: !mo.tensor<[N], si32, dev:0>) -> (!mo.tensor<[1], si32, dev:0>, !mo.tensor<[2], si32, dev:0>, !mo.tensor<[3], si32, dev:0>) {
  %start0 = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[1], si64>
  %stop0 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[1], si64>
  %start1 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[1], si64>
  %stop1 = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[1], si64>
  %start2 = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[1], si64>
  %stop2 = mo.constant <#M.dense_array<6> : tensor<si64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[1], si64>

  %out1 = mo.slice(%input, %start0, %stop0, %step) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si32, dev:0>
  %out2 = mo.slice(%input, %start1, %stop1, %step) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si32, dev:0>
  %out3 = mo.slice(%input, %start2, %stop2, %step) : (!mo.tensor<[N], si32, dev:0>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[3], si32, dev:0>

  %o1 = mo.relu(%out1) : !mo.tensor<[1], si32, dev:0>
  %o2 = mo.relu(%out2) : !mo.tensor<[2], si32, dev:0>
  %o3 = mo.relu(%out3) : !mo.tensor<[3], si32, dev:0>
  mo.output %o1, %o2, %o3 : !mo.tensor<[1], si32, dev:0>, !mo.tensor<[2], si32, dev:0>, !mo.tensor<[3], si32, dev:0>
}

// CHECK-LABEL: mo.graph @slice_to_split_host_constant<N>
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[N], si32, dev:0>
// CHECK-DAG: %[[SPLIT_SIZES:.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[1], si64>
