// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: kgen.func @pop_constant()
kgen.func @pop_constant() {
  // CHECK-NEXT: pop.constant(32 : si64) : !pop.scalar<si64>
  %0 = pop.constant(32 : si64) : !pop.scalar<si64>
  // CHECK-NEXT: pop.constant(3.200000e+01 : f32) : !pop.scalar<f32>
  %1 = pop.constant(32.0 : f32) : !pop.scalar<f32>
  // CHECK-NEXT: pop.constant(3.200000e+01 : f64) : !pop.scalar<f64>
  %2 = pop.constant(32.0 : f64) : !pop.scalar<f64>
  kgen.return
}

// CHECK-LABEL: @pop_constant_simd
kgen.func @pop_constant_simd() {
  // CHECK: pop.constant(#M.dense_array<32, 64>
  %0 = pop.constant(#M.dense_array<32, 64> : vector<2xsi64>) : !pop.simd<2, si64>
  // CHECK: pop.constant(#M.dense_array<3.2{{.*}}, 6.4{{.*}}>
  %1 = pop.constant(#M.dense_array<32., 64.> : vector<2xf64>) : !pop.simd<2, f64>
  // CHECK: pop.constant(#M.dense_array<32, 64>
  %2 = pop.constant(#M.dense_array<32, 64> : vector<2xui32>) : !pop.simd<2, ui32>
  kgen.return
}

// CHECK-LABEL: kgen.generator @pop_constant2<type: dtype>() -> !pop.scalar<type> {
kgen.generator @pop_constant2<type: dtype>() -> !pop.scalar<type> {
  // CHECK-NEXT: pop.constant(32 : i64) : !pop.scalar<type>
  %0 = pop.constant(32) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

// CHECK-LABEL: kgen.func @pop_abs
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_abs(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.abs %[[ARG0]] : !pop.scalar<f32>
  %0 = pop.abs %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_neg
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_neg(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.neg %[[ARG0]] : !pop.scalar<f32>
  %0 = pop.neg %arg0 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_add() -> !pop.scalar<f32> {
kgen.func @pop_add() -> !pop.scalar<f32> {
  // CHECK-NEXT: %[[CST:.*]] = pop.constant(4.000000e+00 : f32) : !pop.scalar<f32>
  %a = pop.constant(4.0 : f32) : !pop.scalar<f32>
  // CHECK-NEXT: %[[CST0:.*]] = pop.constant(6.000000e+00 : f32) : !pop.scalar<f32>
  %b = pop.constant(6.0 : f32) : !pop.scalar<f32>
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[CST]], %[[CST0]] : !pop.scalar<f32>
  %c = pop.add %a, %b : !pop.scalar<f32>
  kgen.return %c : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.generator @pop_add2<type: dtype>
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<type>, %[[ARG1:.*]]: !pop.scalar<type>) -> !pop.scalar<type> {
kgen.generator @pop_add2<type: dtype>(%a: !pop.scalar<type>, %b: !pop.scalar<type>) -> !pop.scalar<type> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !pop.scalar<type>
  %c = pop.add %a, %b : !pop.scalar<type>
  kgen.return %c : !pop.scalar<type>
}

// CHECK-LABEL: kgen.func @pop_add_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_add_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.add %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_sub
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_sub(%arg0 : !pop.scalar<f32>, %arg1 : !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.sub %[[ARG0]], %[[ARG1]] : !pop.scalar<f32>
  %0 = pop.sub %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_sub_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_sub_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.sub %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.sub %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_max
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_max(%arg0 : !pop.scalar<f32>, %arg1 : !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.max %[[ARG0]], %[[ARG1]] : !pop.scalar<f32>
  %0 = pop.max %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_max_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_max_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.max %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.max %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_min
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_min(%arg0 : !pop.scalar<f32>, %arg1 : !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.min %[[ARG0]], %[[ARG1]] : !pop.scalar<f32>
  %0 = pop.min %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_min_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_min_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.min %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.min %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_mul
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_mul(%arg0 : !pop.scalar<f32>, %arg1 : !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.mul %[[ARG0]], %[[ARG1]] : !pop.scalar<f32>
  %0 = pop.mul %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_mul_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_mul_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.mul %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.mul %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_div
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_div(%arg0 : !pop.scalar<f32>, %arg1 : !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.div %[[ARG0]], %[[ARG1]] : !pop.scalar<f32>
  %0 = pop.div %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_div_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_div_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.div %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.div %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @pop_shifts
kgen.func @pop_shifts(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) {
  // CHECK: = pop.shl %{{.*}}, %{{.*}} : !pop.scalar<si32>
  %0 = pop.shl %arg0, %arg1 : !pop.scalar<si32>
  // CHECK: = pop.shr %{{.*}}, %{{.*}} : !pop.scalar<si32>
  %1 = pop.shr %arg0, %arg1 : !pop.scalar<si32>
  kgen.return
}

// CHECK-LABEL: @pop_shifts_simd
kgen.func @pop_shifts_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>) {
  // CHECK: pop.shl %{{.*}}, %{{.*}} : !pop.simd<4, si32>
  %0 = pop.shl %arg0, %arg1 : !pop.simd<4, si32>
  // CHECK: pop.shr %{{.*}}, %{{.*}} : !pop.simd<4, si32>
  %1 = pop.shr %arg0, %arg1 : !pop.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: kgen.func @pop_copysign
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_copysign(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.copysign %[[ARG0]], %[[ARG1]] : !pop.scalar<f32>
  %0 = pop.copysign %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_copysign_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_copysign_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.copysign %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.copysign %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_fma
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>, %[[ARG2:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_fma(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>, %arg2: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.fma %[[ARG0]], %[[ARG1]], %[[ARG2]] : !pop.scalar<f32>
  %0 = pop.fma %arg0, %arg1, %arg2: !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_fma_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>, %[[ARG2:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_fma_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>, %arg2 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.fma %[[ARG0]], %[[ARG1]], %[[ARG2]] : !pop.simd<4, f32>
  %0 = pop.fma %arg0, %arg1, %arg2 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @pop_cmp
kgen.func @pop_cmp(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<bool> {
  // CHECK: pop.cmp ge(%{{.*}}, %{{.*}}) :
  %0 = pop.cmp ge(%arg0, %arg1) : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<bool>
}

kgen.func @pop_cmp_simd(
    %arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>,
    %arg2: !pop.simd<2, f64>, %arg3: !pop.simd<2, f64>
  ) -> (!pop.simd<4, bool>, !pop.simd<2, bool>) {
  // CHECK: pop.cmp ne(%{{.*}}, %{{.*}}) :
  %0 = pop.cmp ne(%arg0, %arg1) : !pop.simd<4, si32>
  // CHECK: pop.cmp lt(%{{.*}}, %{{.*}}) :
  %1 = pop.cmp lt(%arg2, %arg3) : !pop.simd<2, f64>
  kgen.return %0, %1 : !pop.simd<4, bool>, !pop.simd<2, bool>
}


// CHECK-LABEL: @pop_select
kgen.func @pop_select(%arg0 : !pop.scalar<bool>, %arg1: !pop.scalar<f32>, %arg2: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK: pop.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.select %arg0, %arg1, %arg2 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @pop_select_simd
kgen.func @pop_select_simd(
    %arg0: !pop.simd<4, bool>,
    %arg1: !pop.simd<4, si32>,
    %arg2: !pop.simd<4, si32>
  ) -> !pop.simd<4, si32> {
  // CHECK: pop.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.select %arg0, %arg1, %arg2 : !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// CHECK-LABEL: @scalar_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @scalar_bitcast(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f64>) -> !pop.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !pop.scalar<f32> to !pop.scalar<si32>
  %0 = pop.bitcast %arg0 : !pop.scalar<f32> to !pop.scalar<si32>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !pop.scalar<f64> to !pop.scalar<f64>
  %1 = pop.bitcast %arg1 : !pop.scalar<f64> to !pop.scalar<f64>
  // CHECK: %[[V2:.*]] = pop.bitcast %[[V0]] : !pop.scalar<si32> to !pop.scalar<ui32>
  %2 = pop.bitcast %0 : !pop.scalar<si32> to !pop.scalar<ui32>
  // CHECK: %[[V3:.*]] = pop.bitcast %[[V2]] : !pop.scalar<ui32> to !pop.scalar<f32>
  %3 = pop.bitcast %2 : !pop.scalar<ui32> to !pop.scalar<f32>
  // CHECK: return %[[V3]]
  kgen.return %3 : !pop.scalar<f32>
}

// CHECK-LABEL: @simd_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @simd_bitcast(%arg0: !pop.simd<4, f32>, %arg1: !pop.simd<4, f64>) -> !pop.simd<4, f32> {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !pop.simd<4, f32> to !pop.simd<4, si32>
  %0 = pop.bitcast %arg0 : !pop.simd<4, f32> to !pop.simd<4, si32>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !pop.simd<4, f64> to !pop.simd<4, f64>
  %1 = pop.bitcast %arg1 : !pop.simd<4, f64> to !pop.simd<4, f64>
  // CHECK: %[[V2:.*]] = pop.bitcast %[[V0]] : !pop.simd<4, si32> to !pop.simd<4, ui32>
  %2 = pop.bitcast %0 : !pop.simd<4, si32> to !pop.simd<4, ui32>
  // CHECK: %[[V3:.*]] = pop.bitcast %[[V2]] : !pop.simd<4, ui32> to !pop.simd<4, f32>
  %3 = pop.bitcast %2 : !pop.simd<4, ui32> to !pop.simd<4, f32>
  // CHECK: %[[V4:.*]] = pop.bitcast %[[V2]] : !pop.simd<4, ui32> to !pop.simd<2, f64>
  %4 = pop.bitcast %2 : !pop.simd<4, ui32> to !pop.simd<2, f64>
  // CHECK: return %[[V3]]
  kgen.return %3 : !pop.simd<4, f32>
}

// CHECK-LABEL: @bitcast_parametric
kgen.generator @bitcast_parametric<size1, size2, type1: dtype, type2: dtype>(
  %arg0: !pop.simd<size1, type1>, %arg1: !pop.simd<size2, f32>,
  %arg2: !pop.scalar<type2>
) {
  // CHECK: pop.bitcast %{{.*}} : !pop.simd<size1, type1> to !pop.simd<size2, f32>
  %0 = pop.bitcast %arg0 : !pop.simd<size1, type1> to !pop.simd<size2, f32>
  // CHECK: pop.bitcast %{{.*}} : !pop.simd<size2, f32> to !pop.simd<4, f64>
  %1 = pop.bitcast %arg1 : !pop.simd<size2, f32> to !pop.simd<4, f64>
  // CHECK: pop.bitcast %{{.*}} : !pop.scalar<type2> to !pop.scalar<f32>
  %2 = pop.bitcast %arg2 : !pop.scalar<type2> to !pop.scalar<f32>
  kgen.return
}

// CHECK-LABEL: @pointer_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @pointer_bitcast(%arg0: !pop.pointer<!pop.scalar<f32>>, %arg1: !pop.pointer<!pop.simd<4, f64>>) ->
   (!pop.pointer<!pop.simd<4, si32>>, !pop.pointer<!pop.scalar<f64>>) {
  // CHECK: %[[V0:.*]] = pop.pointer.bitcast %[[ARG0]] : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.simd<4, si32>>
  %0 = pop.pointer.bitcast %arg0 : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.simd<4, si32>>
  // CHECK: %[[V1:.*]] = pop.pointer.bitcast %[[ARG1]] : !pop.pointer<!pop.simd<4, f64>> to !pop.pointer<!pop.scalar<f64>>
  %1 = pop.pointer.bitcast %arg1 : !pop.pointer<!pop.simd<4, f64>> to !pop.pointer<!pop.scalar<f64>>
  // CHECK: %{{.*}} = pop.pointer.bitcast %[[ARG0]] : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.scalar<invalid>>
  %2 = pop.pointer.bitcast %arg0 : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.scalar<invalid>>
  // CHECK: return %[[V0]], %[[V1]]
  kgen.return %0, %1 : !pop.pointer<!pop.simd<4, si32>>, !pop.pointer<!pop.scalar<f64>>
}

// CHECK-LABEL: @scalar_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @scalar_cast<type: dtype>(%a: !pop.scalar<f32>) -> !pop.scalar<si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !pop.scalar<f32> to !pop.scalar<type>
  %0 = pop.cast %a : !pop.scalar<f32> to !pop.scalar<type>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !pop.scalar<type> to !pop.scalar<f64>
  %1 = pop.cast %0 : !pop.scalar<type> to !pop.scalar<f64>
  // CHECK: %[[V2:.*]] = pop.cast %[[V1]] : !pop.scalar<f64> to !pop.scalar<si32>
  %2 = pop.cast %1 : !pop.scalar<f64> to !pop.scalar<si32>
  // CHECK: return %[[V2]]
  kgen.return %2 : !pop.scalar<si32>
}

// CHECK-LABEL: @simd_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @simd_cast<size, type: dtype>(%a: !pop.simd<size, f32>) -> !pop.simd<size, si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !pop.simd<size, f32> to !pop.simd<size, type>
  %0 = pop.cast %a : !pop.simd<size, f32> to !pop.simd<size, type>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !pop.simd<size, type> to !pop.simd<size, si32>
  %1 = pop.cast %0 : !pop.simd<size, type> to !pop.simd<size, si32>
  // CHECK: %[[V2:.*]] = pop.cast %[[V1]] : !pop.simd<size, si32> to !pop.simd<size, f64>
  %2 = pop.cast %1 : !pop.simd<size, si32> to !pop.simd<size, f64>
  // CHECK: %[[V3:.*]] = pop.cast %[[V2]] : !pop.simd<size, f64> to !pop.simd<size, si32>
  %3 = pop.cast %2 : !pop.simd<size, f64> to !pop.simd<size, si32>
  // CHECK: return %[[V3]]
  kgen.return %3 : !pop.simd<size, si32>
}

// CHECK-LABEL: @pop_simd_extractelement
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_simd_extractelement<size, type: dtype>(
    %a: !pop.simd<size, type>,
    %b: !pop.simd<size, f32>,
    %c: !pop.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = pop.simd.extractelement %[[A]][%[[IDX]]] : !pop.simd<size, type>
  %u = pop.simd.extractelement %a[%idx] : !pop.simd<size, type>
  // CHECK: %[[V:.*]] = pop.simd.extractelement %[[B]][%[[IDX]]] : !pop.simd<size, f32>
  %v = pop.simd.extractelement %b[%idx] : !pop.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.extractelement %[[C]][%[[IDX]]] : !pop.simd<4, si32>
  %w = pop.simd.extractelement %c[%idx] : !pop.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_simd_insertelement
// CHECK-SAME: %[[V0:[a-z0-9]+]]:
// CHECK-SAME: %[[V1:[a-z0-9]+]]:
// CHECK-SAME: %[[V2:[a-z0-9]+]]:
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_simd_insertelement<size, type: dtype>(
    %v0: !pop.scalar<type>,
    %v1: !pop.scalar<f32>,
    %v2: !pop.scalar<si32>,
    %a: !pop.simd<size, type>,
    %b: !pop.simd<size, f32>,
    %c: !pop.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = pop.simd.insertelement %[[V0]], %[[A]][%[[IDX]]] : !pop.simd<size, type>
  %u = pop.simd.insertelement %v0, %a[%idx] : !pop.simd<size, type>
  // CHECK: %[[V:.*]] = pop.simd.insertelement %[[V1]], %[[B]][%[[IDX]]] : !pop.simd<size, f32>
  %v = pop.simd.insertelement %v1, %b[%idx] : !pop.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.insertelement %[[V2]], %[[C]][%[[IDX]]] : !pop.simd<4, si32>
  %w = pop.simd.insertelement %v2, %c[%idx] : !pop.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_simd_shuffle
kgen.generator @pop_simd_shuffle<size>(%a: !pop.simd<size, f32>, %b: !pop.simd<size, f32>) -> !pop.simd<2, f32> {
  // CHECK: pop.simd.shuffle %{{.*}}, %{{.*}} [1, 2] : !pop.simd<size, f32> -> !pop.simd<2, f32>
  %0 = pop.simd.shuffle %a, %b [1, 2] : !pop.simd<size, f32> -> !pop.simd<2, f32>
  kgen.return %0 : !pop.simd<2, f32>
}

// CHECK-LABEL: @pop_simd_splat
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_splat<size, type: dtype>(%a: !pop.scalar<f32>, %b: !pop.scalar<type>) -> (!pop.simd<4, f32>, !pop.simd<size, type>) {
  // CHECK: %[[U:.*]] = pop.simd.splat %[[A]] : !pop.simd<4, f32>
  %u = pop.simd.splat %a : !pop.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.splat %[[B]] : !pop.simd<size, type>
  %v = pop.simd.splat %b : !pop.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !pop.simd<4, f32>, !pop.simd<size, type>
}

// CHECK-LABEL: @pop_simd_reduce_add
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_reduce_add<size, type: dtype>(%a: !pop.simd<4, f32>, %b: !pop.simd<size, type>) -> (!pop.scalar<f32>, !pop.scalar<type>) {
  // CHECK: %[[U:.*]] = pop.simd.reduce.add %[[A]] : !pop.simd<4, f32>
  %u = pop.simd.reduce.add %a : !pop.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.reduce.add %[[B]] : !pop.simd<size, type>
  %v = pop.simd.reduce.add %b : !pop.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !pop.scalar<f32>, !pop.scalar<type>
}

// CHECK-LABEL: @pop_simd_reduce_mul
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_reduce_mul<size, type: dtype>(%a: !pop.simd<4, f32>, %b: !pop.simd<size, type>) -> (!pop.scalar<f32>, !pop.scalar<type>) {
  // CHECK: %[[U:.*]] = pop.simd.reduce.mul %[[A]] : !pop.simd<4, f32>
  %u = pop.simd.reduce.mul %a : !pop.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.reduce.mul %[[B]] : !pop.simd<size, type>
  %v = pop.simd.reduce.mul %b : !pop.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !pop.scalar<f32>, !pop.scalar<type>
}

// CHECK-LABEL: @pop_simd_reduce_min
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_reduce_min<size, type: dtype>(%a: !pop.simd<4, f32>, %b: !pop.simd<size, type>) -> (!pop.scalar<f32>, !pop.scalar<type>) {
  // CHECK: %[[U:.*]] = pop.simd.reduce.min %[[A]] : !pop.simd<4, f32>
  %u = pop.simd.reduce.min %a : !pop.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.reduce.min %[[B]] : !pop.simd<size, type>
  %v = pop.simd.reduce.min %b : !pop.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !pop.scalar<f32>, !pop.scalar<type>
}

// CHECK-LABEL: @pop_simd_reduce_max
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_reduce_max<size, type: dtype>(%a: !pop.simd<4, f32>, %b: !pop.simd<size, type>) -> (!pop.scalar<f32>, !pop.scalar<type>) {
  // CHECK: %[[U:.*]] = pop.simd.reduce.max %[[A]] : !pop.simd<4, f32>
  %u = pop.simd.reduce.max %a : !pop.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.reduce.max %[[B]] : !pop.simd<size, type>
  %v = pop.simd.reduce.max %b : !pop.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !pop.scalar<f32>, !pop.scalar<type>
}


// CHECK-LABEL: @pop_load_store
kgen.generator @pop_load_store<type: dtype>(%p0: !pop.pointer<!pop.scalar<f32>>, %p1: !pop.pointer<!pop.scalar<type>>) {
  // CHECK: %[[V0:.*]] = pop.load %{{.*}} : !pop.pointer<!pop.scalar<f32>>
  %0 = pop.load %p0 : !pop.pointer<!pop.scalar<f32>>
  // CHECK: %[[V1:.*]] = pop.load %{{.*}} : !pop.pointer<!pop.scalar<type>>
  %1 = pop.load %p1 : !pop.pointer<!pop.scalar<type>>
  // CHECK: pop.store %[[V0]], %{{.*}} : !pop.pointer<!pop.scalar<f32>>
  pop.store %0, %p0 : !pop.pointer<!pop.scalar<f32>>
  // CHECK: pop.store %[[V1]], %{{.*}} : !pop.pointer<!pop.scalar<type>>
  pop.store %1, %p1 : !pop.pointer<!pop.scalar<type>>
  kgen.return
}

// CHECK-LABEL: @pop_prefetch
kgen.generator @pop_prefetch<type: dtype>(%p0: !pop.pointer<!pop.scalar<f32>>) {
  %zero = pop.constant(0 : ui32) : !pop.scalar<ui32>
  // CHECK: pop.prefetch %{{.*}}(NoLocality, ReadDCache) : !pop.pointer<!pop.scalar<f32>>
  pop.prefetch %p0 (NoLocality, ReadDCache) : !pop.pointer<!pop.scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @pop_load_store_alignment
kgen.generator @pop_load_store_alignment<type: dtype>(%p0: !pop.pointer<!pop.scalar<f32>>, %p1: !pop.pointer<!pop.scalar<type>>) {
  // CHECK: pop.load %{{.*}} align 42 : !pop.pointer<!pop.scalar<f32>>
  %0 = pop.load %p0 align 42: !pop.pointer<!pop.scalar<f32>>
  // CHECK: pop.load %{{.*}} align 8 : !pop.pointer<!pop.scalar<type>>
  %1 = pop.load %p1 align 8: !pop.pointer<!pop.scalar<type>>
  // CHECK: pop.store %{{.*}}, %{{.*}} align 4 : !pop.pointer<!pop.scalar<f32>>
  pop.store %0, %p0 align 4: !pop.pointer<!pop.scalar<f32>>
  // CHECK: pop.store %{{.*}}, %{{.*}} align 89 : !pop.pointer<!pop.scalar<type>>
  pop.store %1, %p1 align 89: !pop.pointer<!pop.scalar<type>>
  kgen.return
}

// CHECK-LABEL: @pop_load_alignment_generator
kgen.generator @pop_load_alignment_generator<alignment>(%ptr: !pop.pointer<!pop.scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: pop.load %{{.*}} align alignment : !pop.pointer<!pop.scalar<f32>>
  %0 = pop.load %ptr align alignment : !pop.pointer<!pop.scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @pop_offset
kgen.generator @pop_offset<type: dtype>(%p: !pop.pointer<!pop.scalar<f32>>, %idx: index) {
  // pop.offset %{{.*}}[{{.*}}] : !pop.pointer<!pop.scalar<f32>>
  %0 = pop.offset %p[%idx] : !pop.pointer<!pop.scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @pop_generic_load_store
kgen.generator @pop_generic_load_store<type: type, dtype: dtype, size>(
    %p0: !pop.pointer<type>,
    %p1: !pop.pointer<!pop.scalar<dtype>>,
    %p2: !pop.pointer<!pop.simd<size, dtype>>)
  -> (
    !kgen.paramref<type>,
    !pop.scalar<dtype>,
    !pop.simd<size, dtype>
  ) {
  // CHECK: pop.load %{{.*}} : !pop.pointer<type>
  // CHECK: pop.store %{{.*}} : !pop.pointer<type>
  %0 = pop.load %p0 : !pop.pointer<type>
  pop.store %0, %p0 : !pop.pointer<type>

  // CHECK: pop.load %{{.*}} : !pop.pointer<!pop.scalar<dtype>>
  // CHECK: pop.store %{{.*}} : !pop.pointer<!pop.scalar<dtype>>
  %1 = pop.load %p1 : !pop.pointer<!pop.scalar<dtype>>
  pop.store %1, %p1 : !pop.pointer<!pop.scalar<dtype>>

  // CHECK: pop.load %{{.*}} : !pop.pointer<!pop.simd<size, dtype>>
  // CHECK: pop.store %{{.*}} : !pop.pointer<!pop.simd<size, dtype>>
  %2 = pop.load %p2 : !pop.pointer<!pop.simd<size, dtype>>
  pop.store %2, %p2 : !pop.pointer<!pop.simd<size, dtype>>

  kgen.return %0, %1, %2 : !kgen.paramref<type>, !pop.scalar<dtype>, !pop.simd<size, dtype>
}

// CHECK-LABEL: @pop_generic_offset
kgen.generator @pop_generic_offset<type: type>(
    %p0: !pop.pointer<type>,
    %p1: !pop.pointer<!pop.simd<4, f32>>,
    %i: index) {
  // CHECK: pop.offset %{{.*}} : !pop.pointer<type>
  %0 = pop.offset %p0[%i] : !pop.pointer<type>
  // CHECK: pop.offset %{{.*}} : !pop.pointer<!pop.simd<4, f32>>
  %1 = pop.offset %p1[%i] : !pop.pointer<!pop.simd<4, f32>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @parametricAdd
// CHECK-SAME: %[[ARG0:.*]]: !kgen.paramref<ty>, %[[ARG1:.*]]: !kgen.paramref<ty>
kgen.generator @parametricAdd<ty: type>
  (%arg0: !kgen.paramref<ty>, %arg1: !kgen.paramref<ty>) -> !kgen.paramref<ty> {

  // Fully parametric operations are ok!
  // CHECK: %{{.*}} = pop.add %[[ARG0]], %[[ARG1]] : !kgen.paramref<ty>
  %0 = pop.add %arg0, %arg1 : !kgen.paramref<ty>
  kgen.return %0 : !kgen.paramref<ty>
}

// CHECK-LABEL: @stack_allocation
kgen.generator @stack_allocation<size, type: type>() {
  // CHECK: pop.stack_allocation size : type
  %0 = pop.stack_allocation size : type
  // CHECK: pop.stack_allocation 16 : !pop.simd<4, f32>
  %1 = pop.stack_allocation 16 : !pop.simd<4, f32>
  kgen.return
}

// CHECK-LABEL: @memcpy
// CHECK-SAME: %[[A:.*]]: !pop.pointer<type>
// CHECK-SAME: %[[B:.*]]: !pop.pointer<!pop.scalar<f32>>
// CHECK-SAME: %[[C:.*]]: !pop.pointer<!pop.scalar<si32>>
kgen.generator @memcpy<type: type, dtype: dtype>(%a: !pop.pointer<type>,
                                                 %b: !pop.pointer<!pop.scalar<f32>>,
                                                 %c: !pop.pointer<!pop.scalar<si32>>) {
  // CHECK: %[[SIZE:.*]] = index.constant 1
  %one = index.constant 1
  // CHECK: pop.memcpy %[[A]], %[[A]], %[[SIZE]] : !pop.pointer<type>
  pop.memcpy %a, %a, %one : !pop.pointer<type>
  // CHECK: pop.memcpy inline %[[B]], %[[B]], %[[SIZE]] : !pop.pointer<!pop.scalar<f32>>
  pop.memcpy inline %b, %b, %one : !pop.pointer<!pop.scalar<f32>>
  // CHECK: pop.memcpy inline volatile %[[C]], %[[C]], %[[SIZE]] : !pop.pointer<!pop.scalar<si32>>
  pop.memcpy inline volatile %c, %c, %one : !pop.pointer<!pop.scalar<si32>>
  kgen.return
}


// CHECK-LABEL: @external_call
kgen.generator @external_call<type: type, dtype: dtype>(%a: !kgen.paramref<type>, %b: !pop.scalar<dtype>) {
  // CHECK: pop.external_call @foo(%{{.*}}, %{{.*}})
  %0 = pop.external_call @foo(%a, %b) : (!kgen.paramref<type>, !pop.scalar<dtype>) -> !pop.simd<4, f32>
  // CHECK: pop.external_call @bar(%{{.*}}, %{{.*}}) (!kgen.paramref<type>) -> ()
  pop.external_call @bar(%a, %b) (!kgen.paramref<type>) -> () : (!kgen.paramref<type>, !pop.scalar<dtype>) -> ()
  kgen.return
}

// CHECK-LABEL: @global_constant
kgen.generator @global_constant<type: type, dtype: dtype>() -> !pop.pointer<type> {
  // CHECK: pop.global_constant(5 : i32) : type
  %0 = pop.global_constant(5 : i32) : type
  // CHECK: pop.global_constant(#M.dense_array<0, 1, 2, 3> : !M.array<4xui32>) : !pop.array<4, !pop.scalar<ui32>>
  %1 = pop.global_constant(#M.dense_array<0, 1, 2, 3> : !M.array<4xui32>) : !pop.array<4, !pop.scalar<ui32>>
  // CHECK: pop.global_constant(#M.dense_array<0, 0, 0, 0> : !M.array<4xi32>) : !pop.array<4, !pop.scalar<dtype>>
  %2 = pop.global_constant(#M.dense_array<0, 0, 0, 0> : !M.array<4xi32>) : !pop.array<4, !pop.scalar<dtype>>
  kgen.return %0 : !pop.pointer<type>
}

// CHECK-LABEL: @pointer_to_index
kgen.generator @pointer_to_index<type: type>(%a: !pop.pointer<type>,
                                             %b: !pop.pointer<!pop.scalar<f32>>,
                                             %c: !pop.pointer<!pop.simd<4, f32>>,
                                             %d: !pop.pointer<!pop.scalar<invalid>>) {
  // CHECK: pop.pointer_to_index %{{.*}} : !pop.pointer<type>
  %0 = pop.pointer_to_index %a : !pop.pointer<type>
  // CHECK: pop.pointer_to_index %{{.*}} : !pop.pointer<!pop.scalar<f32>>
  %1 = pop.pointer_to_index %b : !pop.pointer<!pop.scalar<f32>>
  // CHECK: pop.pointer_to_index %{{.*}} : !pop.pointer<!pop.simd<4, f32>>
  %2 = pop.pointer_to_index %c : !pop.pointer<!pop.simd<4, f32>>
  // CHECK: pop.pointer_to_index %{{.*}} : !pop.pointer<!pop.scalar<invalid>>
  %3 = pop.pointer_to_index %d : !pop.pointer<!pop.scalar<invalid>>
  kgen.return
}

// CHECK-LABEL: @index_to_pointer
kgen.generator @index_to_pointer<type: type>(%idx: index) {
  // CHECK: pop.index_to_pointer %{{.*}} : !pop.pointer<type>
  %0 = pop.index_to_pointer %idx : !pop.pointer<type>
  // CHECK: pop.index_to_pointer %{{.*}} : !pop.pointer<!pop.scalar<f32>>
  %1 = pop.index_to_pointer %idx : !pop.pointer<!pop.scalar<f32>>
  // CHECK: pop.index_to_pointer %{{.*}} : !pop.pointer<!pop.simd<4, f32>>
  %2 = pop.index_to_pointer %idx : !pop.pointer<!pop.simd<4, f32>>
  // CHECK: pop.index_to_pointer %{{.*}} : !pop.pointer<!pop.scalar<invalid>>
  %3 = pop.index_to_pointer %idx : !pop.pointer<!pop.scalar<invalid>>
  kgen.return
}

// CHECK-LABEL: @struct
kgen.generator @struct<type: type, dtype: dtype>(
  // CHECK-SAME: %[[A:.*]]: !kgen.paramref
  %a: !kgen.paramref<type>,
  // CHECK-SAME: %[[B:.*]]: !pop.scalar
  %b: !pop.scalar<dtype>
) -> (!kgen.paramref<type>, !pop.scalar<dtype>) {
  // CHECK: %[[S0:.*]] = pop.struct.construct(%[[A]], %[[B]]) : !pop.struct<type, !pop.scalar<dtype>>
  %0 = pop.struct.construct(%a, %b) : !pop.struct<type, !pop.scalar<dtype>>
  // CHECK: %[[V0:.*]] = pop.struct.get %[[S0]][0] : !pop.struct<type, !pop.scalar<dtype>>
  %1 = pop.struct.get %0[0] : !pop.struct<type, !pop.scalar<dtype>>
  // CHECK: %[[V1:.*]] = pop.struct.get %[[S0]][1] : !pop.struct<type, !pop.scalar<dtype>>
  %2 = pop.struct.get %0[1] : !pop.struct<type, !pop.scalar<dtype>>
  // CHECK: pop.struct.replace %{{.*}}, %[[S0]][0] : !pop.struct<type, !pop.scalar<dtype>>
  %3 = pop.struct.replace %1, %0[0] : !pop.struct<type, !pop.scalar<dtype>>
  // CHECK: pop.struct.replace %{{.*}}, %{{.*}}[1] : !pop.struct<type, !pop.scalar<dtype>>
  %4 = pop.struct.replace %2, %3[1] : !pop.struct<type, !pop.scalar<dtype>>
  // CHECK: return %[[V0]], %[[V1]] : !kgen.paramref<type>, !pop.scalar<dtype>
  kgen.return %1, %2 : !kgen.paramref<type>, !pop.scalar<dtype>
}

// CHECK-LABEL: @pointer_types
kgen.generator @pointer_types<dt: dtype>(
  // CHECK-SAME: %{{.*}}: !pop.pointer<!pop.scalar<dt>>, %{{.*}}: !pop.pointer<!pop.scalar<f32>>, %{{.*}}: !pop.pointer<!pop.scalar<invalid>>
  %arg0: !pop.pointer<!pop.scalar<dt>>, %arg1: !pop.pointer<!pop.scalar<f32>>, %arg2: !pop.pointer<!pop.scalar<invalid>>) {
  kgen.return
}

// CHECK-LABEL: @cast_to_builtin
// CHECK-SAME: %[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<si32>
kgen.func @cast_to_builtin(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<si32>) {
  // CHECK: pop.cast_to_builtin %[[ARG0]] : !pop.scalar<f32> to f32
  %0 = pop.cast_to_builtin %arg0: !pop.scalar<f32> to f32
  // CHECK: pop.cast_to_builtin %[[ARG1]] : !pop.scalar<si32> to i32
  %1 = pop.cast_to_builtin %arg1: !pop.scalar<si32> to i32
  kgen.return
}

// CHECK-LABEL: @cast_from_builtin
// CHECK-SAME: %[[ARG0:.*]]: f32, %[[ARG1:.*]]: ui32
kgen.func @cast_from_builtin(%arg0: f32, %arg1: ui32) {
  // CHECK: pop.cast_from_builtin %[[ARG0]] : f32 to !pop.scalar<f32>
  %0 = pop.cast_from_builtin %arg0: f32 to !pop.scalar<f32>
  // CHECK: pop.cast_from_builtin %[[ARG1]] : ui32 to !pop.scalar<ui32>
  %1 = pop.cast_from_builtin %arg1: ui32 to !pop.scalar<ui32>
  kgen.return
}

// CHECK-LABEL: @cast_from_builtin_vector
// CHECK-SAME: %[[ARG:.*]]:
kgen.func @cast_from_builtin_vector(%arg0: vector<1xf32>) -> !pop.simd<1, f32> {
  // CHECK: %[[V0:.*]] = pop.cast_from_builtin %[[ARG]] : vector<1xf32> to !pop.simd<1, f32>
  %0 = pop.cast_from_builtin %arg0 : vector<1xf32> to !pop.simd<1, f32>
  // CHECK: kgen.return  %[[V0:.*]] : !pop.simd<1, f32>
  kgen.return %0 : !pop.simd<1, f32>
}

// CHECK-LABEL: @array_ops
kgen.generator @array_ops<N, T: type, dtype: dtype>(%arg0: !kgen.paramref<T>) -> !pop.array<2, T> {
  // CHECK: pop.array.create [%arg0, %arg0] : !pop.array<2, T>
  %0 = pop.array.create [%arg0, %arg0] : !pop.array<2, T>
  // CHECK: pop.array.get %0[1] : !pop.array<2, T>
  %1 = pop.array.get %0[1] : !pop.array<2, T>
  // CHECK: pop.array.replace %{{.*}}, %0[0] : !pop.array<2, T>
  %2 = pop.array.replace %1, %0[0] : !pop.array<2, T>
  // CHECK: pop.constant(0 : i64) : !pop.array<N, !pop.scalar<dtype>>
  %3 = pop.constant(0) : !pop.array<N, !pop.scalar<dtype>>
  // CHECK: pop.constant(#M.dense_array<0.{{0+}}e+00, 1.{{0+}}e+00, 2.{{0+}}e+00> : !M.array<3xf64>) : !pop.array<3, !pop.scalar<f64>>
  %4 = pop.constant(#M.dense_array<0.0, 1.0, 2.0> : !M.array<3xf64>) : !pop.array<3, !pop.scalar<f64>>
  kgen.return %2 : !pop.array<2, T>
}

// CHECK-LABEL: @variant_type
kgen.generator @variant_type<N, T: type>(%a: !pop.simd<N, f32>) -> !kgen.paramref<T> {
  // CHECK: pop.variant.create %arg0 : !pop.simd<N, f32> -> !pop.variant<T, !pop.simd<N, f32>>
  %0 = pop.variant.create %a : !pop.simd<N, f32> -> !pop.variant<T, !pop.simd<N, f32>>
  // CHECK: pop.variant.is !kgen.paramref<T>, %0 : !pop.variant<T, !pop.simd<N, f32>>
  %1 = pop.variant.is !kgen.paramref<T>, %0 : !pop.variant<T, !pop.simd<N, f32>>
  // CHECK: pop.variant.get %0 : !pop.variant<T, !pop.simd<N, f32>> as !kgen.paramref<T>
  %2 = pop.variant.get %0 : !pop.variant<T, !pop.simd<N, f32>> as !kgen.paramref<T>
  kgen.return %2 : !kgen.paramref<T>
}

// CHECK-LABEL: @variant_canonicalize
// CHECK-SAME: !pop.variant<i32, f32>
kgen.generator.interface @variant_canonicalize(!pop.variant<i32, i32, f32, f32>)

// CHECK-LABEL: @variant_visit
kgen.generator @variant_visit(%a: !pop.variant<i32, f32>) -> index {
  // CHECK: %[[RESULT:.*]] = pop.variant.visit %arg0 : !pop.variant<i32, f32> -> index
  %0 = pop.variant.visit %a : !pop.variant<i32, f32> -> index
  // CHECK-NEXT: case (%[[ARG:.*]]: i32) {
  case (%v: i32) {
    // CHECK: pop.yield %{{.*}} : index
    %1 = index.constant 0
    pop.yield %1 : index
  }
  // CHECK: case (%[[ARG:.*]]: f32) {
  case (%v: f32) {
    %1 = index.constant 1
    // CHECK: pop.yield %{{.*}} : index
    pop.yield %1 : index
  }

  // CHECK: pop.variant.visit %arg0 : !pop.variant<i32, f32>
  pop.variant.visit %a : !pop.variant<i32, f32>
  // CHECK-NEXT: case (%[[ARG:.*]]: f32) {
  case (%v: f32) {
    // CHECK-NEXT: pop.yield
    pop.yield
  }
  // CHECK: default {
  default {
    // CHECK-NEXT: pop.yield
    pop.yield
  }

  // CHECK: return %[[RESULT]]
  kgen.return %0 : index
}

// CHECK-LABEL: @indirect_call
kgen.generator @indirect_call(%a: i32, %fn: (i32) -> index) -> index {
  // CHECK: pop.indirect_call %arg1(%arg0) : (i32) -> index
  %0 = pop.indirect_call %fn(%a) : (i32) -> index
  // CHECK: return %0 : index
  kgen.return %0 : index
}
