// RUN: mo-opt %s --mo-symbolic-optimize | FileCheck %s

// A concat of a consecutive run of one split's results is a view of the
// split's input: one slice over the run, or the input itself when the run is
// every result. The fold runs in the stage where splits are still live, before
// they are lowered back to slices.

// CHECK-LABEL: mo.graph @run_of_results
// CHECK: %[[SLICE:.*]] = mo.slice(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 5], f32>
// CHECK: mo.output %[[SLICE]]
// CHECK-NOT: mo.concat
// CHECK-NOT: mo.split

mo.graph @run_of_results(%x: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 5], f32> {
  %sizes = mo.constant <#M.dense_array<2, 3, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %r:3 = mo.split(%x, %sizes) [axis = 1] : (!mo.tensor<[2, 8], f32>, !mo.tensor<[3], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>)
  %both = mo.concat[1] (%r#0, %r#1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 5], f32>)
  mo.output %both : !mo.tensor<[2, 5], f32>
}

// -----

// CHECK-LABEL: mo.graph @every_result
// CHECK: mo.output %arg0
// CHECK-NOT: mo.concat
// CHECK-NOT: mo.split
// CHECK-NOT: mo.slice

mo.graph @every_result(%x: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 8], f32> {
  %sizes = mo.constant <#M.dense_array<2, 3, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %r:3 = mo.split(%x, %sizes) [axis = 1] : (!mo.tensor<[2, 8], f32>, !mo.tensor<[3], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>)
  %all = mo.concat[1] (%r#0, %r#1, %r#2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 8], f32>)
  mo.output %all : !mo.tensor<[2, 8], f32>
}
