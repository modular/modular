// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @select(%[[COND:.*]]: !mo.tensor<[2, 3], bool>, %[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @select(%cond: !mo.tensor<[2, 3], bool>, %x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.select(%[[COND]], %[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], bool>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.select(%cond, %x, %y) : (!mo.tensor<[2, 3], bool>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @select_int_folder() -> !mo.tensor<[2, 3], si32> {
mo.graph @select_int_folder() -> (!mo.tensor<[2, 3], si32>) {
  // CHECK: %[[RES:.*]] = mo.constant <#M.dense_array<1, 8, 9, 4, 11, 6> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %cond = mo.constant <#M.dense_array<1, 0, 0, 1, 0, 1> : tensor<2x3xi8>> : !mo.tensor<[2, 3], bool>
  %then = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %else = mo.constant <#M.dense_array<7, 8, 9, 10, 11, 12> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %res = mo.select(%cond, %then, %else) : (
    !mo.tensor<[2, 3], bool>, !mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>
  ) -> !mo.tensor<[2, 3], si32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], si32>
  mo.output %res : !mo.tensor<[2, 3], si32>
}
