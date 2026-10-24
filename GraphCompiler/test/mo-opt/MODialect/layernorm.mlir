// RUN: mo-opt -mo-shape-inference -mo-symbolic-optimize -mo-pattern-fusion %s | FileCheck %s --check-prefix=CHECK-IR

// CHECK-IR: mo.graph @tf_layer_norm(%[[ARG0:.*]]:  !mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 2], f32>
mo.graph @tf_layer_norm(%arg0 : !mo.tensor<[4, 3, 2], f32>) -> (!mo.tensor<[4, 3, 2], f32>) {
  // CHECK-IR: %[[EPS:.*]] = mo.constant {{.*}} : !mo.tensor<[], f32>
  // CHECK-IR: %[[GAM:.*]] = mo.constant <#M.dense_array<1.100{{.*}}> : !mo.tensor<[2], f32>
  // CHECK-IR: %[[BET:.*]] = mo.constant <#M.dense_array<-1.000{{.*}}> : !mo.tensor<[2], f32>
  // CHECK-IR: %[[VAL1:.*]] = mo.reduce.layer_norm(%[[ARG0]], %[[GAM]], %[[BET]], %[[EPS]])
  // CHECK-IR: mo.output %[[VAL1]]
  %sh = "mo.constant"() {value = #M.dense_array<4, 3, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 3, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %v1_sum = mo.reduce.add(%arg0) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v1_len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v1_len_b = mo.static.broadcast_to(%v1_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %1 = mo.div(%v1_sum, %v1_len_b) : !mo.tensor<[4, 3, 1], f32>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %3 = mo.sub(%arg0, %2) : !mo.tensor<[4, 3, 2], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 3, 2], f32>
  %v5_sum = mo.reduce.add(%4) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v5_len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v5_len_b = mo.static.broadcast_to(%v5_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %5 = mo.div(%v5_sum, %v5_len_b) : !mo.tensor<[4, 3, 1], f32>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 3, 1], f32>
  %8 = mo.rsqrt(%7) : !mo.tensor<[4, 3, 1], f32>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %10 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %11 = mo.mul(%9, %10) : !mo.tensor<[4, 3, 2], f32>
  %12 = mo.mul(%arg0, %11) : !mo.tensor<[4, 3, 2], f32>
  %13 = mo.mul(%2, %11) : !mo.tensor<[4, 3, 2], f32>
  %14 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %15 = mo.sub(%14, %13) : !mo.tensor<[4, 3, 2], f32>
  %16 = mo.add(%12, %15) : !mo.tensor<[4, 3, 2], f32>
  mo.output %16 : !mo.tensor<[4, 3, 2], f32>
}

// CHECK-IR: mo.graph @tf_layer_norm_skip(%[[ARG0:.*]]:  !mo.tensor<[4, 3, 2], f32>, %[[ARG1:.*]]: !mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 2], f32>
mo.graph @tf_layer_norm_skip(%arg0 : !mo.tensor<[4, 3, 2], f32>, %arg1 : !mo.tensor<[4, 3, 2], f32>) -> (!mo.tensor<[4, 3, 2], f32>) {
  // CHECK-IR: %[[EPS:.*]] = mo.constant {{.*}} : !mo.tensor<[], f32>
  // CHECK-IR: %[[GAM:.*]] = mo.constant <#M.dense_array<1.100{{.*}}> : !mo.tensor<[2], f32>
  // CHECK-IR: %[[BET:.*]] = mo.constant <#M.dense_array<-1.000{{.*}}> : !mo.tensor<[2], f32>
  // CHECK-IR: %[[VAL:.*]] = mo.add(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[4, 3, 2], f32>
  // CHECK-IR: %[[VAL1:.*]] = mo.reduce.layer_norm(%[[VAL]], %[[GAM]], %[[BET]], %[[EPS]])
  // CHECK-IR: mo.output %[[VAL1]]
  %sh = "mo.constant"() {value = #M.dense_array<4, 3, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 3, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[4, 3, 2], f32>
  %v1_sum = mo.reduce.add(%0) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v1_len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v1_len_b = mo.static.broadcast_to(%v1_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %1 = mo.div(%v1_sum, %v1_len_b) : !mo.tensor<[4, 3, 1], f32>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %3 = mo.sub(%0, %2) : !mo.tensor<[4, 3, 2], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 3, 2], f32>
  %v5_sum = mo.reduce.add(%4) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v5_len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v5_len_b = mo.static.broadcast_to(%v5_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %5 = mo.div(%v5_sum, %v5_len_b) : !mo.tensor<[4, 3, 1], f32>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 3, 1], f32>
  %8 = mo.rsqrt(%7) : !mo.tensor<[4, 3, 1], f32>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %10 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %11 = mo.mul(%9, %10) : !mo.tensor<[4, 3, 2], f32>
  %12 = mo.mul(%0, %11) : !mo.tensor<[4, 3, 2], f32>
  %13 = mo.mul(%2, %11) : !mo.tensor<[4, 3, 2], f32>
  %14 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %15 = mo.sub(%14, %13) : !mo.tensor<[4, 3, 2], f32>
  %16 = mo.add(%12, %15) : !mo.tensor<[4, 3, 2], f32>
  mo.output %16 : !mo.tensor<[4, 3, 2], f32>
}

// CHECK-IR: mo.graph @pt_layer_norm(%[[ARG0:.*]]:  !mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 2], f32>
mo.graph @pt_layer_norm(%arg0 : !mo.tensor<[4, 3, 2], f32>) -> (!mo.tensor<[4, 3, 2], f32>) {
  // CHECK-IR: %[[EPS:.*]] = mo.constant {{.*}} : !mo.tensor<[], f32>
  // CHECK-IR: %[[GAM:.*]] = mo.constant <#M.dense_array<1.100{{.*}}> : !mo.tensor<[2], f32>
  // CHECK-IR: %[[BET:.*]] = mo.constant <#M.dense_array<-1.000{{.*}}> : !mo.tensor<[2], f32>
  // CHECK-IR: %[[VAL1:.*]] = mo.reduce.layer_norm(%[[ARG0]], %[[GAM]], %[[BET]], %[[EPS]])
  // CHECK-IR: mo.output %[[VAL1]]
  %sh = "mo.constant"() {value = #M.dense_array<4, 3, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 3, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %v1_sum = mo.reduce.add(%arg0) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v1_len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v1_len_b = mo.static.broadcast_to(%v1_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %1 = mo.div(%v1_sum, %v1_len_b) : !mo.tensor<[4, 3, 1], f32>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %3 = mo.sub(%arg0, %2) : !mo.tensor<[4, 3, 2], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 3, 2], f32>
  %v5_sum = mo.reduce.add(%4) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v5_len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v5_len_b = mo.static.broadcast_to(%v5_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %5 = mo.div(%v5_sum, %v5_len_b) : !mo.tensor<[4, 3, 1], f32>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 3, 1], f32>
  %8 = mo.sqrt(%7) : !mo.tensor<[4, 3, 1], f32>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %10 = mo.div(%3, %9) : !mo.tensor<[4, 3, 2], f32>
  %11 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %12 = mo.mul(%10, %11) : !mo.tensor<[4, 3, 2], f32>
  %13 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %14 = mo.add(%12, %13) : !mo.tensor<[4, 3, 2], f32>
  mo.output %14 : !mo.tensor<[4, 3, 2], f32>
}

// A wide last dim lets fast-math rewrite the division by sqrt into a
// multiplication by rsqrt before fusion; the pattern must still fuse.
// CHECK-IR: mo.graph @pt_layer_norm_wide<[[B:.*]]>(%[[ARG0:.*]]: !mo.tensor<[[[B]], 3, 16], f32>) -> !mo.tensor<[[[B]], 3, 16], f32>
mo.graph @pt_layer_norm_wide<B>(%arg0 : !mo.tensor<[B, 3, 16], f32>) -> (!mo.tensor<[B, 3, 16], f32>) {
  // CHECK-IR: %[[EPS:.*]] = mo.constant {{.*}} : !mo.tensor<[], f32>
  // CHECK-IR: %[[GAM:.*]] = mo.constant <#M.dense_array<1.100{{.*}}> : !mo.tensor<[16], f32>
  // CHECK-IR: %[[BET:.*]] = mo.constant <#M.dense_array<-1.000{{.*}}> : !mo.tensor<[16], f32>
  // CHECK-IR: %[[VAL1:.*]] = mo.reduce.layer_norm(%[[ARG0]], %[[GAM]], %[[BET]], %[[EPS]])
  // CHECK-IR: mo.output %[[VAL1]]
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1, 1.1, 2.1, 1.1, 2.1, 1.1, 2.1, 1.1, 2.1, 1.1, 2.1, 1.1, 2.1, 1.1, 2.1> : tensor<16xf32>} : () -> !mo.tensor<[16], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0> : tensor<16xf32>} : () -> !mo.tensor<[16], f32>
  %v1_sum = mo.reduce.add(%arg0) : (!mo.tensor<[B, 3, 16], f32>) -> !mo.tensor<[B, 3, 1], f32>
  %v1_len = mo.constant <#M.dense_array<16.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v1_len_b = mo.static.broadcast_to(%v1_len) : !mo.tensor<[], f32> -> !mo.tensor<[B, 3, 1], f32>
  %1 = mo.div(%v1_sum, %v1_len_b) : !mo.tensor<[B, 3, 1], f32>
  %2 = mo.static.broadcast_to(%1) : !mo.tensor<[B, 3, 1], f32> -> !mo.tensor<[B, 3, 16], f32>
  %3 = mo.sub(%arg0, %2) : !mo.tensor<[B, 3, 16], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[B, 3, 16], f32>
  %v5_sum = mo.reduce.add(%4) : (!mo.tensor<[B, 3, 16], f32>) -> !mo.tensor<[B, 3, 1], f32>
  %v5_len = mo.constant <#M.dense_array<16.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v5_len_b = mo.static.broadcast_to(%v5_len) : !mo.tensor<[], f32> -> !mo.tensor<[B, 3, 1], f32>
  %5 = mo.div(%v5_sum, %v5_len_b) : !mo.tensor<[B, 3, 1], f32>
  %6 = mo.static.broadcast_to(%eps) : !mo.tensor<[], f32> -> !mo.tensor<[B, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[B, 3, 1], f32>
  %8 = mo.sqrt(%7) : !mo.tensor<[B, 3, 1], f32>
  %9 = mo.static.broadcast_to(%8) : !mo.tensor<[B, 3, 1], f32> -> !mo.tensor<[B, 3, 16], f32>
  %10 = mo.div(%3, %9) : !mo.tensor<[B, 3, 16], f32>
  %11 = mo.static.broadcast_to(%gamma) : !mo.tensor<[16], f32> -> !mo.tensor<[B, 3, 16], f32>
  %12 = mo.mul(%10, %11) : !mo.tensor<[B, 3, 16], f32>
  %13 = mo.static.broadcast_to(%beta) : !mo.tensor<[16], f32> -> !mo.tensor<[B, 3, 16], f32>
  %14 = mo.add(%12, %13) : !mo.tensor<[B, 3, 16], f32>
  mo.output %14 : !mo.tensor<[B, 3, 16], f32>
}

// A mean that divides by something other than the axis length isn't a layer
// norm: nothing fuses.
// CHECK-IR-LABEL: mo.graph @pt_layer_norm_wrong_divisor
// CHECK-IR-NOT:   mo.reduce.layer_norm
// CHECK-IR:       mo.output
mo.graph @pt_layer_norm_wrong_divisor(%arg0 : !mo.tensor<[4, 3, 2], f32>) -> (!mo.tensor<[4, 3, 2], f32>) {
  %sh = "mo.constant"() {value = #M.dense_array<4, 3, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 3, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %v1_sum = mo.reduce.add(%arg0) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v1_len = mo.constant <#M.dense_array<3.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v1_len_b = mo.static.broadcast_to(%v1_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %1 = mo.div(%v1_sum, %v1_len_b) : !mo.tensor<[4, 3, 1], f32>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %3 = mo.sub(%arg0, %2) : !mo.tensor<[4, 3, 2], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 3, 2], f32>
  %v5_sum = mo.reduce.add(%4) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v5_len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v5_len_b = mo.static.broadcast_to(%v5_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %5 = mo.div(%v5_sum, %v5_len_b) : !mo.tensor<[4, 3, 1], f32>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 3, 1], f32>
  %8 = mo.sqrt(%7) : !mo.tensor<[4, 3, 1], f32>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %10 = mo.div(%3, %9) : !mo.tensor<[4, 3, 2], f32>
  %11 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %12 = mo.mul(%10, %11) : !mo.tensor<[4, 3, 2], f32>
  %13 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %14 = mo.add(%12, %13) : !mo.tensor<[4, 3, 2], f32>
  mo.output %14 : !mo.tensor<[4, 3, 2], f32>
}

// Means that multiply by an already folded reciprocal of the axis length.
// CHECK-IR-LABEL: mo.graph @pt_layer_norm_folded_reciprocal
// CHECK-IR:       mo.reduce.layer_norm
mo.graph @pt_layer_norm_folded_reciprocal(%arg0 : !mo.tensor<[4, 3, 2], f32>) -> (!mo.tensor<[4, 3, 2], f32>) {
  %sh = "mo.constant"() {value = #M.dense_array<4, 3, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 3, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %v1_sum = mo.reduce.add(%arg0) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v1_len = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v1_len_b = mo.static.broadcast_to(%v1_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %1 = mo.mul(%v1_sum, %v1_len_b) : !mo.tensor<[4, 3, 1], f32>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %3 = mo.sub(%arg0, %2) : !mo.tensor<[4, 3, 2], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 3, 2], f32>
  %v5_sum = mo.reduce.add(%4) : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %v5_len = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
  %v5_len_b = mo.static.broadcast_to(%v5_len) : !mo.tensor<[], f32> -> !mo.tensor<[4, 3, 1], f32>
  %5 = mo.mul(%v5_sum, %v5_len_b) : !mo.tensor<[4, 3, 1], f32>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 3, 1], f32>
  %8 = mo.sqrt(%7) : !mo.tensor<[4, 3, 1], f32>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %10 = mo.div(%3, %9) : !mo.tensor<[4, 3, 2], f32>
  %11 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %12 = mo.mul(%10, %11) : !mo.tensor<[4, 3, 2], f32>
  %13 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %14 = mo.add(%12, %13) : !mo.tensor<[4, 3, 2], f32>
  mo.output %14 : !mo.tensor<[4, 3, 2], f32>
}

// The same in f16, where the folded reciprocal of 3 is 1/3 rounded to f16.
// CHECK-IR-LABEL: mo.graph @pt_layer_norm_folded_reciprocal_f16
// CHECK-IR:       mo.reduce.layer_norm
mo.graph @pt_layer_norm_folded_reciprocal_f16(%arg0 : !mo.tensor<[4, 2, 3], f16>) -> (!mo.tensor<[4, 2, 3], f16>) {
  %sh = "mo.constant"() {value = #M.dense_array<4, 2, 3> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 2, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f16>} : () -> !mo.tensor<[], f16>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1, 3.1> : tensor<3xf16>} : () -> !mo.tensor<[3], f16>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0, -3.0> : tensor<3xf16>} : () -> !mo.tensor<[3], f16>
  %v1_sum = mo.reduce.add(%arg0) : (!mo.tensor<[4, 2, 3], f16>) -> !mo.tensor<[4, 2, 1], f16>
  %v1_len = mo.constant <#M.dense_array<0.333251953125> : tensor<1xf16>> : !mo.tensor<[], f16>
  %v1_len_b = mo.static.broadcast_to(%v1_len) : !mo.tensor<[], f16> -> !mo.tensor<[4, 2, 1], f16>
  %1 = mo.mul(%v1_sum, %v1_len_b) : !mo.tensor<[4, 2, 1], f16>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 2, 1], f16>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 2, 3], f16>
  %3 = mo.sub(%arg0, %2) : !mo.tensor<[4, 2, 3], f16>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 2, 3], f16>
  %v5_sum = mo.reduce.add(%4) : (!mo.tensor<[4, 2, 3], f16>) -> !mo.tensor<[4, 2, 1], f16>
  %v5_len = mo.constant <#M.dense_array<0.333251953125> : tensor<1xf16>> : !mo.tensor<[], f16>
  %v5_len_b = mo.static.broadcast_to(%v5_len) : !mo.tensor<[], f16> -> !mo.tensor<[4, 2, 1], f16>
  %5 = mo.mul(%v5_sum, %v5_len_b) : !mo.tensor<[4, 2, 1], f16>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f16>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 2, 1], f16>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 2, 1], f16>
  %8 = mo.sqrt(%7) : !mo.tensor<[4, 2, 1], f16>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 2, 1], f16>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 2, 3], f16>
  %10 = mo.div(%3, %9) : !mo.tensor<[4, 2, 3], f16>
  %11 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[3], f16>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 2, 3], f16>
  %12 = mo.mul(%10, %11) : !mo.tensor<[4, 2, 3], f16>
  %13 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[3], f16>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 2, 3], f16>
  %14 = mo.add(%12, %13) : !mo.tensor<[4, 2, 3], f16>
  mo.output %14 : !mo.tensor<[4, 2, 3], f16>
}
