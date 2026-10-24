// RUN: mo-opt %s --mo-constant-fold | FileCheck %s

// CHECK: mo.graph @reduce_mul_fold() -> !mo.tensor<[1], si64> {
mo.graph @reduce_mul_fold() -> !mo.tensor<[1], si64> {
  %input = mo.constant <#M.dense_array<2, 3, 4> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.constant <#M.dense_array<24> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
  %output = mo.reduce.mul(%input) [axis = 0] : (!mo.tensor<[3], si64>) -> !mo.tensor<[1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[1], si64>
  mo.output %output : !mo.tensor<[1], si64>
}

// CHECK: mo.graph @add_folder()
mo.graph @add_folder() -> (!mo.tensor<[2, 2], si32>) {
  // CHECK: %[[VAL0:.*]] = mo.constant <#M.dense_array<-1, 2, 0, 4> : !M.array<4xsi32>> : !mo.tensor<[2, 2], si32>
  %0 = mo.constant <#M.dense_array<-1, 0, 1, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %1 = mo.constant <#M.dense_array<0, 2, -1, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %2 = mo.add(%0, %1) : !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: mo.output %[[VAL0]] : !mo.tensor<[2, 2], si32>
  mo.output %2 : !mo.tensor<[2, 2], si32>
}

mo.graph @fold_div() -> (!mo.tensor<[7], si32>) {
    %0 = "mo.constant"() {value = #M.dense_array<1, -5, 10, 3, 1, 250, -343> : tensor<7xsi32>} : () -> !mo.tensor<[7], si32>
    %1 = "mo.constant"() {value = #M.dense_array<1, 5, -2, 2, 3, 5, -343> : tensor<7xsi32>} : () -> !mo.tensor<[7], si32>

    %output = mo.div(%0, %1) : !mo.tensor<[7], si32>
    // CHECK-LABEL: mo.graph @fold_div
    // CHECK-NEXT: %[[CST:.*]] = mo.constant <#M.dense_array<1, -1, -5, 1, 0, 50, 1> : !M.array<7xsi32>>
    // CHECK-NEXT: mo.output %[[CST]]
    mo.output %output : !mo.tensor<[7], si32>
}

// CHECK: mo.graph @equal_int_folder() -> !mo.tensor<[2, 3], bool> {
mo.graph @equal_int_folder() -> (!mo.tensor<[2, 3], bool>) {
  // CHECK: %[[RES:.*]] = mo.constant <#M.dense_array<false, true, false, false, false, true> : !M.array<6xi1>> : !mo.tensor<[2, 3], bool>
  %lhs = mo.constant <#M.dense_array<0, -1, 2, -2, 4, 5> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %rhs = mo.constant <#M.dense_array<2, -1, 1, 4, -4, 5> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %0 = mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], bool>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @greater_int_folder() -> !mo.tensor<[2, 3], bool> {
mo.graph @greater_int_folder() -> (!mo.tensor<[2, 3], bool>) {
  // CHECK: %[[RES:.*]] = mo.constant <#M.dense_array<false, false, true, false, true, false> : !M.array<6xi1>> : !mo.tensor<[2, 3], bool>
  %lhs = mo.constant <#M.dense_array<0, -1, 2, -2, 4, 5> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %rhs = mo.constant <#M.dense_array<2, -1, 1, 4, -4, 5> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %0 = mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], bool>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @greater_equal_int_folder() -> !mo.tensor<[2, 3], bool> {
mo.graph @greater_equal_int_folder() -> (!mo.tensor<[2, 3], bool>) {
  // CHECK: %[[RES:.*]] = mo.constant <#M.dense_array<false, true, true, false, true, true> : !M.array<6xi1>> : !mo.tensor<[2, 3], bool>
  %lhs = mo.constant <#M.dense_array<0, -1, 2, -2, 4, 5> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %rhs = mo.constant <#M.dense_array<2, -1, 1, 4, -4, 5> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %0 = mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], bool>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @not_equal_int_folder() -> !mo.tensor<[2, 3], bool> {
mo.graph @not_equal_int_folder() -> (!mo.tensor<[2, 3], bool>) {
  // CHECK: %[[RES:.*]] = mo.constant <#M.dense_array<true, false, true, true, true, false> : !M.array<6xi1>> : !mo.tensor<[2, 3], bool>
  %lhs = mo.constant <#M.dense_array<0, -1, 2, -2, 4, 5> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %rhs = mo.constant <#M.dense_array<2, -1, 1, 4, -4, 5> : tensor<2x3xsi32>> : !mo.tensor<[2, 3], si32>
  %0 = mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], bool>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @broadcast_shape_fold() -> !mo.tensor<[3], si32> {
mo.graph @broadcast_shape_fold() -> !mo.tensor<[3], si32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.constant <#M.dense_array<5, 10, 42> : !M.array<3xsi32>> : !mo.tensor<[3], si32>
  %s0 = mo.constant <#M.dense_array<5, 10, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %s1 = mo.constant <#M.dense_array<10, 42> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %res = mo.broadcast_shape(%s0, %s1) : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[3], si32>
  mo.output %res : !mo.tensor<[3], si32>
}

// COM: Do not fold broadcasts as outputs in constant-folding
// CHECK: mo.graph @static_broadcast_nofold_constant_inputs() -> !mo.tensor<[2, 2], f32> {
mo.graph @static_broadcast_nofold_constant_inputs() -> !mo.tensor<[2, 2], f32> {
  %from = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
  // CHECK: [[OUT:%.*]] = mo.static.broadcast_to
  // CHECK-NEXT: mo.output [[OUT]]
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[1], f32> -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @gather_empty_input_folder
mo.graph @gather_empty_input_folder<D0>(%indices: !mo.tensor<[D0], si32>) -> !mo.tensor<[0], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.constant <#M.dense_array<> : tensor<0xsi64>> : !mo.tensor<[0], si64>
  %0 = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  %1 = mo.gather(%0, %indices) [axis = 0] : (!mo.tensor<[0], si64>, !mo.tensor<[D0], si32>) -> !mo.tensor<[0], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[0], si64>
  mo.output %1 : !mo.tensor<[0], si64>
}

// CHECK: mo.graph @gather_param_no_fold<[[D0:.*]] -> [[D1:.*]]>(
mo.graph @gather_param_no_fold<D0 -> D1>(
    // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]]], si32>) -> !mo.tensor<[[[D1]]], si32> {
    %input: !mo.tensor<[D0], si32>) -> !mo.tensor<[D1], si32> {
  // CHECK-DAG: %[[INDICES:.*]] = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  %indices = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.gather<() -> [[D2:.*]]>(%[[INPUT]], %[[INDICES]]) [axis = 0] : (!mo.tensor<[D0], si32>, !mo.tensor<[0], si64>) -> !mo.tensor<[[[D2]]], si32>
  %1 = mo.gather<() -> D2>(%input, %indices) [axis = 0] : (!mo.tensor<[D0], si32>, !mo.tensor<[0], si64>) -> !mo.tensor<[D2], si32>
  // CHECK-NEXT: mo.output<[[D2]]> %[[RES]] : !mo.tensor<[[[D2]]], si32>
  mo.output<D2> %1 : !mo.tensor<[D2], si32>
}

// CHECK: mo.graph @gather_empty_input_index
mo.graph @gather_empty_input_index<D0>(%input: !mo.tensor<[D0], si32>) -> !mo.tensor<[0], si32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.constant <#M.dense_array<> : tensor<0xsi32>> : !mo.tensor<[0], si32>
  %indices = mo.constant <#M.dense_array<> : !M.array<0xsi64>> : !mo.tensor<[0], si64>
  %1 = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[D0], si32>, !mo.tensor<[0], si64>) -> !mo.tensor<[0], si32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[0], si32>
  mo.output %1 : !mo.tensor<[0], si32>
}

// CHECK-LABEL: mo.graph @fold_gather
// CHECK-NEXT: %[[CST:.*]] = mo.constant <#M.dense_array<22, 33, 44>
// CHECK-NEXT: mo.output %[[CST]]
mo.graph @fold_gather() -> !mo.tensor<[3], si64> {
    %data = mo.constant <#M.dense_array<0, 11, 22, 33, 44, 55> : tensor<6xsi64>> : !mo.tensor<[6], si64>
    %indices = mo.constant <#M.dense_array<2, 3, 4> : tensor<3xsi32>> : !mo.tensor<[3], si32>
    %gather = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[6], si64>, !mo.tensor<[3], si32>) -> !mo.tensor<[3], si64>
    mo.output %gather : !mo.tensor<[3], si64>
}

// CHECK-LABEL: mo.graph @no_fold_gather_dyn_indices
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.gather
// CHECK-NEXT: mo.output %[[OUT]]
mo.graph @no_fold_gather_dyn_indices(%indices : !mo.tensor<[3], si32>) -> !mo.tensor<[3], si64> {
    %data = mo.constant <#M.dense_array<0, 11, 22, 33, 44, 55> : tensor<6xsi64>> : !mo.tensor<[6], si64>
    %gather = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[6], si64>, !mo.tensor<[3], si32>) -> !mo.tensor<[3], si64>
    mo.output %gather : !mo.tensor<[3], si64>
}

// CHECK-LABEL: mo.graph @no_fold_gather_dyn_input
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.gather
// CHECK-NEXT: mo.output %[[OUT]]
mo.graph @no_fold_gather_dyn_input(%data : !mo.tensor<[6], si64>) -> !mo.tensor<[3], si64> {
    %indices = mo.constant <#M.dense_array<2, 3, 4> : tensor<3xsi32>> : !mo.tensor<[3], si32>
    %gather = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[6], si64>, !mo.tensor<[3], si32>) -> !mo.tensor<[3], si64>
    mo.output %gather : !mo.tensor<[3], si64>
}
// CHECK-LABEL: mo.graph @no_fold_elided_constant
// CHECK-NEXT: mo.constant
// CHECK-NEXT: mo.constant
// CHECK-NEXT: mo.static.broadcast_to
// CHECK-NEXT: %[[OUT:.*]] = mo.greater
// CHECK-NEXT: mo.output %[[OUT]]
mo.graph @no_fold_elided_constant() -> !mo.tensor<[64, 256], bool> {
    %zero = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
    %weight = mo.constant <dense_resource<weight> : tensor<64x256xf32>> : !mo.tensor<[64, 256], f32>
    %res = mo.static.broadcast_to(%zero) : !mo.tensor<[], f32> -> !mo.tensor<[64, 256], f32>
    %out = mo.greater(%weight, %res) : (!mo.tensor<[64, 256], f32>,
                                     !mo.tensor<[64, 256], f32>
                                    ) -> !mo.tensor<[64, 256], bool>
    mo.output %out : !mo.tensor<[64, 256], bool>
}

{-#

#-}
