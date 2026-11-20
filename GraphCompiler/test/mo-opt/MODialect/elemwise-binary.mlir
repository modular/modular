// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @add(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @add(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.add(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @div(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @div(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.div(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.div(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @max(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @max(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.max(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.max(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @min(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @min(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.min(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.min(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @mul(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @mul(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.mul(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.mul(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @pow(%[[ARG0:.*]]: ![[ARG0TYPE:.*]], %[[ARG1:.*]]: ![[ARG1TYPE:.*]]) -> !mo.tensor<[2, 3], f32> {
mo.graph @pow(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.pow(%[[ARG0]] : ![[ARG0TYPE]], %[[ARG1]] :
  %0 = mo.pow(%arg0 : !mo.tensor<[2, 3], f32>, %arg1 : !mo.tensor<[2, 3], f32>) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @pow_int_exponent(%[[ARG0:.*]]: ![[ARG0TYPE:.*]], %[[ARG1:.*]]: ![[ARG1TYPE:.*]]) -> !mo.tensor<[2, 3], f32> {
mo.graph @pow_int_exponent(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], si32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.pow(%[[ARG0]] : ![[ARG0TYPE]], %[[ARG1]] : ![[ARG1TYPE]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.pow(%arg0 : !mo.tensor<[2, 3], f32>, %arg1 : !mo.tensor<[2, 3], si32>) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @sub(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @sub(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.sub(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.sub(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @equal(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool> {
mo.graph @equal(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.equal(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  %0 = mo.equal(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @greater(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool> {
mo.graph @greater(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.greater(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  %0 = mo.greater(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @greater_equal(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool> {
mo.graph @greater_equal(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.greater_equal(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  %0 = mo.greater_equal(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @not_equal(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool> {
mo.graph @not_equal(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.not_equal(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  %0 = mo.not_equal(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @and(%[[ARG0:.*]]: !mo.tensor<[2, 3], bool>, %[[ARG1:.*]]: !mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
mo.graph @and(%arg0: !mo.tensor<[2, 3], bool>, %arg1: !mo.tensor<[2, 3], bool>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.and(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], bool>
  %0 = mo.and(%arg0, %arg1) : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @xor(%[[ARG0:.*]]: !mo.tensor<[2, 3], bool>, %[[ARG1:.*]]: !mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
mo.graph @xor(%arg0: !mo.tensor<[2, 3], bool>, %arg1: !mo.tensor<[2, 3], bool>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.xor(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], bool>
  %0 = mo.xor(%arg0, %arg1) : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @mod(%[[ARG0:.*]]: !mo.tensor<[2, 3], si32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], si32>
mo.graph @mod(%arg0: !mo.tensor<[2, 3], si32>, %arg1: !mo.tensor<[2, 3], si32>) -> (!mo.tensor<[2, 3], si32>) {
  // CHECK-NEXT: mo.mod(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], si32>
  %0 = mo.mod(%arg0, %arg1) : !mo.tensor<[2, 3], si32>
  mo.output %0 : !mo.tensor<[2, 3], si32>
}

// CHECK-LABEL: mo.graph @mul_by_one_fold<N, K, L>
// CHECK: (%{{.*}}: !mo.tensor<[N, N], bool>, %[[ARG1:.*]]: !mo.tensor<[L, L], bool>,
// CHECK-NEXT: mo.output %[[ARG1]], %[[ARG1]]

mo.graph @mul_by_one_fold<N, K, L>(%arg0: !mo.tensor<[N, N], bool>, %arg1: !mo.tensor<[L, L], bool>, %arg2: !mo.tensor<[2], si64>) -> (!mo.tensor<[L, L], bool>, !mo.tensor<[L, L], bool>) {
  %0 = mo.constant <#M.dense_array<true> : tensor<i1>> : !mo.tensor<[], bool>
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[], bool> -> !mo.tensor<[N, N], bool>
  %2 = mo.slice(%1, %arg2, %arg2, %arg2) : (!mo.tensor<[N, N], bool>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[K, K], bool>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.transpose(%2, %perm) : (!mo.tensor<[K, K], bool>, !mo.tensor<[2], si64>) -> !mo.tensor<[K, K], bool>
  %4 = mo.rebind(%3) : !mo.tensor<[K, K], bool> -> !mo.tensor<[L, L], bool>
  %5 = mo.mul(%4, %arg1) : !mo.tensor<[L, L], bool>
  %6 = mo.mul(%arg1, %4) : !mo.tensor<[L, L], bool>
  mo.output %5, %6 : !mo.tensor<[L, L], bool>, !mo.tensor<[L, L], bool>
}


// CHECK-LABEL: mo.graph @mul_by_one_foldf32<N, K, L>
// CHECK: (%{{.*}}: !mo.tensor<[N, N], f32>, %[[ARG1:.*]]: !mo.tensor<[L, L], f32>,
// CHECK-NEXT: mo.output %[[ARG1]], %[[ARG1]]

mo.graph @mul_by_one_foldf32<N, K, L>(%arg0: !mo.tensor<[N, N], f32>, %arg1: !mo.tensor<[L, L], f32>, %arg2: !mo.tensor<[2], si64>) -> (!mo.tensor<[L, L], f32>, !mo.tensor<[L, L], f32>) {
  %0 = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32> -> !mo.tensor<[N, N], f32>
  %2 = mo.slice(%1, %arg2, %arg2, %arg2) : (!mo.tensor<[N, N], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[K, K], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.transpose(%2, %perm) : (!mo.tensor<[K, K], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[K, K], f32>
  %4 = mo.rebind(%3) : !mo.tensor<[K, K], f32> -> !mo.tensor<[L, L], f32>
  %5 = mo.mul(%4, %arg1) : !mo.tensor<[L, L], f32>
  %6 = mo.mul(%arg1, %4) : !mo.tensor<[L, L], f32>
  mo.output %5, %6 : !mo.tensor<[L, L], f32>, !mo.tensor<[L, L], f32>
}

// CHECK-LABEL: mo.graph @mul_by_one_foldsi64<N, K, L>
// CHECK: (%{{.*}}: !mo.tensor<[N, N], si64>, %[[ARG1:.*]]: !mo.tensor<[L, L], si64>,
// CHECK-NEXT: mo.output %[[ARG1]], %[[ARG1]]

mo.graph @mul_by_one_foldsi64<N, K, L>(%arg0: !mo.tensor<[N, N], si64>, %arg1: !mo.tensor<[L, L], si64>, %arg2: !mo.tensor<[2], si64>) -> (!mo.tensor<[L, L], si64>, !mo.tensor<[L, L], si64>) {
  %0 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[], si64> -> !mo.tensor<[N, N], si64>
  %2 = mo.slice(%1, %arg2, %arg2, %arg2) : (!mo.tensor<[N, N], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[K, K], si64>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.transpose(%2, %perm) : (!mo.tensor<[K, K], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[K, K], si64>
  %4 = mo.rebind(%3) : !mo.tensor<[K, K], si64> -> !mo.tensor<[L, L], si64>
  %5 = mo.mul(%4, %arg1) : !mo.tensor<[L, L], si64>
  %6 = mo.mul(%arg1, %4) : !mo.tensor<[L, L], si64>
  mo.output %5, %6 : !mo.tensor<[L, L], si64>, !mo.tensor<[L, L], si64>
}
