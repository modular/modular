// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// Matrix Band
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @linalg_band_part() -> !mo.tensor<[1, 2, 3], f32>
mo.graph @linalg_band_part() -> !mo.tensor<[1, 2, 3], f32> {
  // CHECK: %[[VAL_0:.*]] = mo.constant <#M.dense_array<1.{{0+}}e+00, 2.{{0+}}e+00, 3.{{0+}}e+00, 4.{{0+}}e+00, 5.{{0+}}e+00, 6.{{0+}}e+00> : tensor<1x2x3xf32>> : !mo.tensor<[1, 2, 3], f32>
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<1x2x3xf32>> : !mo.tensor<[1, 2, 3], f32>
  // CHECK: %[[VAL_1:.*]] = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  %1 = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  // CHECK: %[[VAL_2:.*]] = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  // CHECK: %[[VAL_3:.*]] = mo.constant <#M.dense_array<0> : tensor<i8>> : !mo.tensor<[], bool>
  %3 = mo.constant <#M.dense_array<0> : tensor<i8>> : !mo.tensor<[], bool>
  // CHECK: %[[VAL_4:.*]] = mo.linalg.band_part(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]])
  %4 = mo.linalg.band_part(%0, %1, %2, %3) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[1, 2, 3], f32>
  // CHECK: mo.output %[[VAL_4]] : !mo.tensor<[1, 2, 3], f32>
  mo.output %4 : !mo.tensor<[1, 2, 3], f32>
}

// CHECK: mo.graph @linalg_band_part_dynamic_batch(%[[VAL_0:.*]]: !mo.tensor<[?, 2, 3], f32>) -> !mo.tensor<[?, 2, 3], f32>
mo.graph @linalg_band_part_dynamic_batch(%arg0: !mo.tensor<[?, 2, 3], f32>) -> !mo.tensor<[?, 2, 3], f32> {
  // CHECK: %[[VAL_1:.*]] = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  %1 = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>
  // CHECK: %[[VAL_2:.*]] = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %2 = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  // CHECK: %[[VAL_3:.*]] = mo.constant <#M.dense_array<0> : tensor<i8>> : !mo.tensor<[], bool>
  %3 = mo.constant <#M.dense_array<0> : tensor<i8>> : !mo.tensor<[], bool>
  // CHECK: %[[VAL_4:.*]] = mo.linalg.band_part(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]])
  %4 = mo.linalg.band_part(%arg0, %1, %2, %3) : (!mo.tensor<[?, 2, 3], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[?, 2, 3], f32>
  // CHECK: mo.output %[[VAL_4]] : !mo.tensor<[?, 2, 3], f32>
  mo.output %4 : !mo.tensor<[?, 2, 3], f32>
}

