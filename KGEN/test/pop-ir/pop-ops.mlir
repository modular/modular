// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: kgen.func @pop_constant()
kgen.func @pop_constant() {
  // CHECK-NEXT: pop.constant(32 : si64) : !meta.scalar<si64>
  %0 = pop.constant(32 : si64) : !meta.scalar<si64>
  // CHECK-NEXT: pop.constant(3.200000e+01 : f32) : !meta.scalar<f32>
  %1 = pop.constant(32.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: pop.constant(3.200000e+01 : f64) : !meta.scalar<f64>
  %2 = pop.constant(32.0 : f64) : !meta.scalar<f64>
  kgen.return
}

// CHECK-LABEL: @pop_constant_simd
kgen.func @pop_constant_simd() {
  // CHECK: pop.constant(dense<[32, 64]>
  %0 = pop.constant(dense<[32, 64]> : vector<2xsi64>) : !meta.simd<2, si64>
  // CHECK: pop.constant(dense<[3.2{{.*}}, 6.4{{.*}}]>
  %1 = pop.constant(dense<[32., 64.]> : vector<2xf64>) : !meta.simd<2, f64>
  // CHECK: pop.constant(dense<[32, 64]>
  %2 = pop.constant(dense<[32, 64]> : vector<2xui32>) : !meta.simd<2, ui32>
  kgen.return
}

// CHECK-LABEL: kgen.generator @pop_constant2<type: dtype>() -> !meta.scalar<type> {
kgen.generator @pop_constant2<type: dtype>() -> !meta.scalar<type> {
  // CHECK-NEXT: pop.constant(32 : i64) : !meta.scalar<type>
  %0 = pop.constant(32) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// CHECK-LABEL: kgen.func @pop_abs
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_abs(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.abs %[[ARG0]] : !meta.scalar<f32>
  %0 = pop.abs %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_neg
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_neg(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.neg %[[ARG0]] : !meta.scalar<f32>
  %0 = pop.neg %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_add() -> !meta.scalar<f32> {
kgen.func @pop_add() -> !meta.scalar<f32> {
  // CHECK-NEXT: %[[CST:.*]] = pop.constant(4.000000e+00 : f32) : !meta.scalar<f32>
  %a = pop.constant(4.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %[[CST0:.*]] = pop.constant(6.000000e+00 : f32) : !meta.scalar<f32>
  %b = pop.constant(6.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[CST]], %[[CST0]] : !meta.scalar<f32>
  %c = pop.add %a, %b : !meta.scalar<f32>
  kgen.return %c : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.generator @pop_add2<type: dtype>
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<type>, %[[ARG1:.*]]: !meta.scalar<type>) -> !meta.scalar<type> {
kgen.generator @pop_add2<type: dtype>(%a: !meta.scalar<type>, %b: !meta.scalar<type>) -> !meta.scalar<type> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !meta.scalar<type>
  %c = pop.add %a, %b : !meta.scalar<type>
  kgen.return %c : !meta.scalar<type>
}

// CHECK-LABEL: kgen.func @pop_add_simd
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<4, f32>, %[[ARG1:.*]]: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.func @pop_add_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !meta.simd<4, f32>
  %0 = pop.add %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_sub
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_sub(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.sub %[[ARG0]], %[[ARG1]] : !meta.scalar<f32>
  %0 = pop.sub %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_sub_simd
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<4, f32>, %[[ARG1:.*]]: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.func @pop_sub_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.sub %[[ARG0]], %[[ARG1]] : !meta.simd<4, f32>
  %0 = pop.sub %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_max
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_max(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.max %[[ARG0]], %[[ARG1]] : !meta.scalar<f32>
  %0 = pop.max %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_max_simd
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<4, f32>, %[[ARG1:.*]]: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.func @pop_max_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.max %[[ARG0]], %[[ARG1]] : !meta.simd<4, f32>
  %0 = pop.max %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_min
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_min(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.min %[[ARG0]], %[[ARG1]] : !meta.scalar<f32>
  %0 = pop.min %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_min_simd
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<4, f32>, %[[ARG1:.*]]: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.func @pop_min_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.min %[[ARG0]], %[[ARG1]] : !meta.simd<4, f32>
  %0 = pop.min %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_mul
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_mul(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.mul %[[ARG0]], %[[ARG1]] : !meta.scalar<f32>
  %0 = pop.mul %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_mul_simd
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<4, f32>, %[[ARG1:.*]]: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.func @pop_mul_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.mul %[[ARG0]], %[[ARG1]] : !meta.simd<4, f32>
  %0 = pop.mul %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_div
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_div(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.div %[[ARG0]], %[[ARG1]] : !meta.scalar<f32>
  %0 = pop.div %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_div_simd
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<4, f32>, %[[ARG1:.*]]: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.func @pop_div_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.div %[[ARG0]], %[[ARG1]] : !meta.simd<4, f32>
  %0 = pop.div %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: @pop_shifts
kgen.func @pop_shifts(%arg0: !meta.scalar<si32>, %arg1: !meta.scalar<si32>) {
  // CHECK: = pop.shl %{{.*}}, %{{.*}} : !meta.scalar<si32>
  %0 = pop.shl %arg0, %arg1 : !meta.scalar<si32>
  // CHECK: = pop.shr %{{.*}}, %{{.*}} : !meta.scalar<si32>
  %1 = pop.shr %arg0, %arg1 : !meta.scalar<si32>
  kgen.return
}

// CHECK-LABEL: @pop_shifts_simd
kgen.func @pop_shifts_simd(%arg0: !meta.simd<4, si32>, %arg1: !meta.simd<4, si32>) {
  // CHECK: pop.shl %{{.*}}, %{{.*}} : !meta.simd<4, si32>
  %0 = pop.shl %arg0, %arg1 : !meta.simd<4, si32>
  // CHECK: pop.shr %{{.*}}, %{{.*}} : !meta.simd<4, si32>
  %1 = pop.shr %arg0, %arg1 : !meta.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: kgen.func @pop_copysign
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_copysign(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.copysign %[[ARG0]], %[[ARG1]] : !meta.scalar<f32>
  %0 = pop.copysign %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_copysign_simd
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<4, f32>, %[[ARG1:.*]]: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.func @pop_copysign_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.copysign %[[ARG0]], %[[ARG1]] : !meta.simd<4, f32>
  %0 = pop.copysign %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_fma
// CHECK-SAME: (%[[ARG0:.*]]: !meta.scalar<f32>, %[[ARG1:.*]]: !meta.scalar<f32>, %[[ARG2:.*]]: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.func @pop_fma(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.fma %[[ARG0]], %[[ARG1]], %[[ARG2]] : !meta.scalar<f32>
  %0 = pop.fma %arg0, %arg1, %arg2: !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_fma_simd
// CHECK-SAME: (%[[ARG0:.*]]: !meta.simd<4, f32>, %[[ARG1:.*]]: !meta.simd<4, f32>, %[[ARG2:.*]]: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.func @pop_fma_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>, %arg2 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.fma %[[ARG0]], %[[ARG1]], %[[ARG2]] : !meta.simd<4, f32>
  %0 = pop.fma %arg0, %arg1, %arg2 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: @pop_cmp
kgen.func @pop_cmp(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<bool> {
  // CHECK: pop.cmp ge(%{{.*}}, %{{.*}}) :
  %0 = pop.cmp ge(%arg0, %arg1) : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<bool>
}

kgen.func @pop_cmp_simd(
    %arg0: !meta.simd<4, si32>, %arg1: !meta.simd<4, si32>,
    %arg2: !meta.simd<2, f64>, %arg3: !meta.simd<2, f64>
  ) -> (!meta.simd<4, bool>, !meta.simd<2, bool>) {
  // CHECK: pop.cmp ne(%{{.*}}, %{{.*}}) :
  %0 = pop.cmp ne(%arg0, %arg1) : !meta.simd<4, si32>
  // CHECK: pop.cmp lt(%{{.*}}, %{{.*}}) :
  %1 = pop.cmp lt(%arg2, %arg3) : !meta.simd<2, f64>
  kgen.return %0, %1 : !meta.simd<4, bool>, !meta.simd<2, bool>
}


// CHECK-LABEL: @pop_select
kgen.func @pop_select(%arg0 : !meta.scalar<bool>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: pop.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: @pop_select_simd
kgen.func @pop_select_simd(
    %arg0: !meta.simd<4, bool>,
    %arg1: !meta.simd<4, si32>,
    %arg2: !meta.simd<4, si32>
  ) -> !meta.simd<4, si32> {
  // CHECK: pop.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.simd<4, si32>
  kgen.return %0 : !meta.simd<4, si32>
}

// CHECK-LABEL: @scalar_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @scalar_bitcast(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f64>) -> !meta.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !meta.scalar<f32> to !meta.scalar<si32>
  %0 = pop.bitcast %arg0 : !meta.scalar<f32> to !meta.scalar<si32>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !meta.scalar<f64> to !meta.scalar<f64>
  %1 = pop.bitcast %arg1 : !meta.scalar<f64> to !meta.scalar<f64>
  // CHECK: %[[V2:.*]] = pop.bitcast %[[V0]] : !meta.scalar<si32> to !meta.scalar<ui32>
  %2 = pop.bitcast %0 : !meta.scalar<si32> to !meta.scalar<ui32>
  // CHECK: %[[V3:.*]] = pop.bitcast %[[V2]] : !meta.scalar<ui32> to !meta.scalar<f32>
  %3 = pop.bitcast %2 : !meta.scalar<ui32> to !meta.scalar<f32>
  // CHECK: return %[[V3]]
  kgen.return %3 : !meta.scalar<f32>
}

// CHECK-LABEL: @simd_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @simd_bitcast(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f64>) -> !meta.simd<4, f32> {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !meta.simd<4, f32> to !meta.simd<4, si32>
  %0 = pop.bitcast %arg0 : !meta.simd<4, f32> to !meta.simd<4, si32>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !meta.simd<4, f64> to !meta.simd<4, f64>
  %1 = pop.bitcast %arg1 : !meta.simd<4, f64> to !meta.simd<4, f64>
  // CHECK: %[[V2:.*]] = pop.bitcast %[[V0]] : !meta.simd<4, si32> to !meta.simd<4, ui32>
  %2 = pop.bitcast %0 : !meta.simd<4, si32> to !meta.simd<4, ui32>
  // CHECK: %[[V3:.*]] = pop.bitcast %[[V2]] : !meta.simd<4, ui32> to !meta.simd<4, f32>
  %3 = pop.bitcast %2 : !meta.simd<4, ui32> to !meta.simd<4, f32>
  // CHECK: %[[V4:.*]] = pop.bitcast %[[V2]] : !meta.simd<4, ui32> to !meta.simd<2, f64>
  %4 = pop.bitcast %2 : !meta.simd<4, ui32> to !meta.simd<2, f64>
  // CHECK: return %[[V3]]
  kgen.return %3 : !meta.simd<4, f32>
}

// CHECK-LABEL: @bitcast_parametric
kgen.generator @bitcast_parametric<size1, size2, type1: dtype, type2: dtype>(
  %arg0: !meta.simd<size1, type1>, %arg1: !meta.simd<size2, f32>,
  %arg2: !meta.scalar<type2>
) {
  // CHECK: pop.bitcast %{{.*}} : !meta.simd<size1, type1> to !meta.simd<size2, f32>
  %0 = pop.bitcast %arg0 : !meta.simd<size1, type1> to !meta.simd<size2, f32>
  // CHECK: pop.bitcast %{{.*}} : !meta.simd<size2, f32> to !meta.simd<4, f64>
  %1 = pop.bitcast %arg1 : !meta.simd<size2, f32> to !meta.simd<4, f64>
  // CHECK: pop.bitcast %{{.*}} : !meta.scalar<type2> to !meta.scalar<f32>
  %2 = pop.bitcast %arg2 : !meta.scalar<type2> to !meta.scalar<f32>
  kgen.return
}

// CHECK-LABEL: @pointer_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @pointer_bitcast(%arg0: !meta.pointer<!meta.scalar<f32>>, %arg1: !meta.pointer<!meta.simd<4, f64>>) ->
   (!meta.pointer<!meta.simd<4, si32>>, !meta.pointer<!meta.scalar<f64>>) {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !meta.pointer<!meta.scalar<f32>> to !meta.pointer<!meta.simd<4, si32>>
  %0 = pop.bitcast %arg0 : !meta.pointer<!meta.scalar<f32>> to !meta.pointer<!meta.simd<4, si32>>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !meta.pointer<!meta.simd<4, f64>> to !meta.pointer<!meta.scalar<f64>>
  %1 = pop.bitcast %arg1 : !meta.pointer<!meta.simd<4, f64>> to !meta.pointer<!meta.scalar<f64>>
  // CHECK: %{{.*}} = pop.bitcast %[[ARG0]] : !meta.pointer<!meta.scalar<f32>> to !meta.pointer<?>
  %2 = pop.bitcast %arg0 : !meta.pointer<!meta.scalar<f32>> to !meta.pointer<?>
  // CHECK: return %[[V0]], %[[V1]]
  kgen.return %0, %1 : !meta.pointer<!meta.simd<4, si32>>, !meta.pointer<!meta.scalar<f64>>
}

// CHECK-LABEL: @scalar_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @scalar_cast<type: dtype>(%a: !meta.scalar<f32>) -> !meta.scalar<si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !meta.scalar<f32> to !meta.scalar<type>
  %0 = pop.cast %a : !meta.scalar<f32> to !meta.scalar<type>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !meta.scalar<type> to !meta.scalar<f64>
  %1 = pop.cast %0 : !meta.scalar<type> to !meta.scalar<f64>
  // CHECK: %[[V2:.*]] = pop.cast %[[V1]] : !meta.scalar<f64> to !meta.scalar<si32>
  %2 = pop.cast %1 : !meta.scalar<f64> to !meta.scalar<si32>
  // CHECK: return %[[V2]]
  kgen.return %2 : !meta.scalar<si32>
}

// CHECK-LABEL: @simd_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @simd_cast<size, type: dtype>(%a: !meta.simd<size, f32>) -> !meta.simd<size, si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !meta.simd<size, f32> to !meta.simd<size, type>
  %0 = pop.cast %a : !meta.simd<size, f32> to !meta.simd<size, type>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !meta.simd<size, type> to !meta.simd<size, si32>
  %1 = pop.cast %0 : !meta.simd<size, type> to !meta.simd<size, si32>
  // CHECK: %[[V2:.*]] = pop.cast %[[V1]] : !meta.simd<size, si32> to !meta.simd<size, f64>
  %2 = pop.cast %1 : !meta.simd<size, si32> to !meta.simd<size, f64>
  // CHECK: %[[V3:.*]] = pop.cast %[[V2]] : !meta.simd<size, f64> to !meta.simd<size, si32>
  %3 = pop.cast %2 : !meta.simd<size, f64> to !meta.simd<size, si32>
  // CHECK: return %[[V3]]
  kgen.return %3 : !meta.simd<size, si32>
}

// CHECK-LABEL: @pop_simd_extractelement
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_simd_extractelement<size, type: dtype>(
    %a: !meta.simd<size, type>,
    %b: !meta.simd<size, f32>,
    %c: !meta.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = pop.simd.extractelement %[[A]][%[[IDX]]] : !meta.simd<size, type>
  %u = pop.simd.extractelement %a[%idx] : !meta.simd<size, type>
  // CHECK: %[[V:.*]] = pop.simd.extractelement %[[B]][%[[IDX]]] : !meta.simd<size, f32>
  %v = pop.simd.extractelement %b[%idx] : !meta.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.extractelement %[[C]][%[[IDX]]] : !meta.simd<4, si32>
  %w = pop.simd.extractelement %c[%idx] : !meta.simd<4, si32>
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
    %v0: !meta.scalar<type>,
    %v1: !meta.scalar<f32>,
    %v2: !meta.scalar<si32>,
    %a: !meta.simd<size, type>,
    %b: !meta.simd<size, f32>,
    %c: !meta.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = pop.simd.insertelement %[[V0]], %[[A]][%[[IDX]]] : !meta.simd<size, type>
  %u = pop.simd.insertelement %v0, %a[%idx] : !meta.simd<size, type>
  // CHECK: %[[V:.*]] = pop.simd.insertelement %[[V1]], %[[B]][%[[IDX]]] : !meta.simd<size, f32>
  %v = pop.simd.insertelement %v1, %b[%idx] : !meta.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.insertelement %[[V2]], %[[C]][%[[IDX]]] : !meta.simd<4, si32>
  %w = pop.simd.insertelement %v2, %c[%idx] : !meta.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_simd_shuffle
kgen.generator @pop_simd_shuffle<size>(%a: !meta.simd<size, f32>, %b: !meta.simd<size, f32>) -> !meta.simd<2, f32> {
  // CHECK: pop.simd.shuffle %{{.*}}, %{{.*}} [1, 2] : !meta.simd<size, f32> -> !meta.simd<2, f32>
  %0 = pop.simd.shuffle %a, %b [1, 2] : !meta.simd<size, f32> -> !meta.simd<2, f32>
  kgen.return %0 : !meta.simd<2, f32>
}

// CHECK-LABEL: @pop_simd_splat
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_splat<size, type: dtype>(%a: !meta.scalar<f32>, %b: !meta.scalar<type>) -> (!meta.simd<4, f32>, !meta.simd<size, type>) {
  // CHECK: %[[U:.*]] = pop.simd.splat %[[A]] : !meta.simd<4, f32>
  %u = pop.simd.splat %a : !meta.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.splat %[[B]] : !meta.simd<size, type>
  %v = pop.simd.splat %b : !meta.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !meta.simd<4, f32>, !meta.simd<size, type>
}

// CHECK-LABEL: @pop_simd_reduce_add
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_reduce_add<size, type: dtype>(%a: !meta.simd<4, f32>, %b: !meta.simd<size, type>) -> (!meta.scalar<f32>, !meta.scalar<type>) {
  // CHECK: %[[U:.*]] = pop.simd.reduce.add %[[A]] : !meta.simd<4, f32>
  %u = pop.simd.reduce.add %a : !meta.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.reduce.add %[[B]] : !meta.simd<size, type>
  %v = pop.simd.reduce.add %b : !meta.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !meta.scalar<f32>, !meta.scalar<type>
}

// CHECK-LABEL: @pop_simd_reduce_mul
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_reduce_mul<size, type: dtype>(%a: !meta.simd<4, f32>, %b: !meta.simd<size, type>) -> (!meta.scalar<f32>, !meta.scalar<type>) {
  // CHECK: %[[U:.*]] = pop.simd.reduce.mul %[[A]] : !meta.simd<4, f32>
  %u = pop.simd.reduce.mul %a : !meta.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.reduce.mul %[[B]] : !meta.simd<size, type>
  %v = pop.simd.reduce.mul %b : !meta.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !meta.scalar<f32>, !meta.scalar<type>
}

// CHECK-LABEL: @pop_simd_reduce_min
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_reduce_min<size, type: dtype>(%a: !meta.simd<4, f32>, %b: !meta.simd<size, type>) -> (!meta.scalar<f32>, !meta.scalar<type>) {
  // CHECK: %[[U:.*]] = pop.simd.reduce.min %[[A]] : !meta.simd<4, f32>
  %u = pop.simd.reduce.min %a : !meta.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.reduce.min %[[B]] : !meta.simd<size, type>
  %v = pop.simd.reduce.min %b : !meta.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !meta.scalar<f32>, !meta.scalar<type>
}

// CHECK-LABEL: @pop_simd_reduce_max
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_reduce_max<size, type: dtype>(%a: !meta.simd<4, f32>, %b: !meta.simd<size, type>) -> (!meta.scalar<f32>, !meta.scalar<type>) {
  // CHECK: %[[U:.*]] = pop.simd.reduce.max %[[A]] : !meta.simd<4, f32>
  %u = pop.simd.reduce.max %a : !meta.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.reduce.max %[[B]] : !meta.simd<size, type>
  %v = pop.simd.reduce.max %b : !meta.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !meta.scalar<f32>, !meta.scalar<type>
}


// CHECK-LABEL: @pop_load_store
kgen.generator @pop_load_store<type: dtype>(%p0: !meta.pointer<!meta.scalar<f32>>, %p1: !meta.pointer<!meta.scalar<type>>) {
  // CHECK: %[[V0:.*]] = pop.load %{{.*}} : !meta.pointer<!meta.scalar<f32>>
  %0 = pop.load %p0 : !meta.pointer<!meta.scalar<f32>>
  // CHECK: %[[V1:.*]] = pop.load %{{.*}} : !meta.pointer<!meta.scalar<type>>
  %1 = pop.load %p1 : !meta.pointer<!meta.scalar<type>>
  // CHECK: pop.store %[[V0]], %{{.*}} : !meta.pointer<!meta.scalar<f32>>
  pop.store %0, %p0 : !meta.pointer<!meta.scalar<f32>>
  // CHECK: pop.store %[[V1]], %{{.*}} : !meta.pointer<!meta.scalar<type>>
  pop.store %1, %p1 : !meta.pointer<!meta.scalar<type>>
  kgen.return
}

// CHECK-LABEL: @pop_offset
kgen.generator @pop_offset<type: dtype>(%p: !meta.pointer<!meta.scalar<f32>>, %idx: index) {
  // pop.offset %{{.*}}[{{.*}}] : !meta.pointer<!meta.scalar<f32>>
  %0 = pop.offset %p[%idx] : !meta.pointer<!meta.scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @pop_generic_load_store
kgen.generator @pop_generic_load_store<type: type, dtype: dtype, size>(
    %p0: !meta.pointer<type>,
    %p1: !meta.pointer<!meta.scalar<dtype>>,
    %p2: !meta.pointer<!meta.simd<size, dtype>>)
  -> (
    !kgen.paramref<type>,
    !meta.scalar<dtype>,
    !meta.simd<size, dtype>
  ) {
  // CHECK: pop.load %{{.*}} : !meta.pointer<type>
  // CHECK: pop.store %{{.*}} : !meta.pointer<type>
  %0 = pop.load %p0 : !meta.pointer<type>
  pop.store %0, %p0 : !meta.pointer<type>

  // CHECK: pop.load %{{.*}} : !meta.pointer<!meta.scalar<dtype>>
  // CHECK: pop.store %{{.*}} : !meta.pointer<!meta.scalar<dtype>>
  %1 = pop.load %p1 : !meta.pointer<!meta.scalar<dtype>>
  pop.store %1, %p1 : !meta.pointer<!meta.scalar<dtype>>

  // CHECK: pop.load %{{.*}} : !meta.pointer<!meta.simd<size, dtype>>
  // CHECK: pop.store %{{.*}} : !meta.pointer<!meta.simd<size, dtype>>
  %2 = pop.load %p2 : !meta.pointer<!meta.simd<size, dtype>>
  pop.store %2, %p2 : !meta.pointer<!meta.simd<size, dtype>>

  kgen.return %0, %1, %2 : !kgen.paramref<type>, !meta.scalar<dtype>, !meta.simd<size, dtype>
}

// CHECK-LABEL: @pop_generic_offset
kgen.generator @pop_generic_offset<type: type>(
    %p0: !meta.pointer<type>,
    %p1: !meta.pointer<!meta.simd<4, f32>>,
    %i: index) {
  // CHECK: pop.offset %{{.*}} : !meta.pointer<type>
  %0 = pop.offset %p0[%i] : !meta.pointer<type>
  // CHECK: pop.offset %{{.*}} : !meta.pointer<!meta.simd<4, f32>>
  %1 = pop.offset %p1[%i] : !meta.pointer<!meta.simd<4, f32>>
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
  // CHECK: pop.stack_allocation 16 : !meta.simd<4, f32>
  %1 = pop.stack_allocation 16 : !meta.simd<4, f32>
  kgen.return
}

// CHECK-LABEL: @external_call
kgen.generator @external_call<type: type, dtype: dtype>(%a: !kgen.paramref<type>, %b: !meta.scalar<dtype>) {
  // CHECK: pop.external_call @foo(%{{.*}}, %{{.*}})
  %0 = pop.external_call @foo(%a, %b) : (!kgen.paramref<type>, !meta.scalar<dtype>) -> !meta.simd<4, f32>
  // CHECK: pop.external_call @bar(%{{.*}}, %{{.*}}) (!kgen.paramref<type>) -> ()
  pop.external_call @bar(%a, %b) (!kgen.paramref<type>) -> () : (!kgen.paramref<type>, !meta.scalar<dtype>) -> ()
  kgen.return
}

// CHECK-LABEL: @global_constant
kgen.generator @global_constant<type: type, dtype: dtype>() -> !meta.pointer<type> {
  // CHECK: pop.global_constant(5 : i32) : type
  %0 = pop.global_constant(5 : i32) : type
  // CHECK: pop.global_constant(dense<[0, 1, 2, 3]> : tensor<4xui32>) : !pop.array<4, !meta.scalar<ui32>>
  %1 = pop.global_constant(dense<[0, 1, 2, 3]> : tensor<4xui32>) : !pop.array<4, !meta.scalar<ui32>>
  // CHECK: pop.global_constant(dense<0> : tensor<4xi32>) : !pop.array<4, !meta.scalar<dtype>>
  %2 = pop.global_constant(dense<0> : tensor<4xi32>) : !pop.array<4, !meta.scalar<dtype>>
  kgen.return %0 : !meta.pointer<type>
}

// CHECK-LABEL: @pointer_to_index
kgen.generator @pointer_to_index<type: type>(%a: !meta.pointer<type>,
                                             %b: !meta.pointer<!meta.scalar<f32>>,
                                             %c: !meta.pointer<!meta.simd<4, f32>>,
                                             %d: !meta.pointer<?>) {
  // CHECK: pop.pointer_to_index %{{.*}} : !meta.pointer<type>
  %0 = pop.pointer_to_index %a : !meta.pointer<type>
  // CHECK: pop.pointer_to_index %{{.*}} : !meta.pointer<!meta.scalar<f32>>
  %1 = pop.pointer_to_index %b : !meta.pointer<!meta.scalar<f32>>
  // CHECK: pop.pointer_to_index %{{.*}} : !meta.pointer<!meta.simd<4, f32>>
  %2 = pop.pointer_to_index %c : !meta.pointer<!meta.simd<4, f32>>
  // CHECK: pop.pointer_to_index %{{.*}} : !meta.pointer<?>
  %3 = pop.pointer_to_index %d : !meta.pointer<?>
  kgen.return
}

// CHECK-LABEL: @index_to_pointer
kgen.generator @index_to_pointer<type: type>(%idx: index) {
  // CHECK: pop.index_to_pointer %{{.*}} : !meta.pointer<type>
  %0 = pop.index_to_pointer %idx : !meta.pointer<type>
  // CHECK: pop.index_to_pointer %{{.*}} : !meta.pointer<!meta.scalar<f32>>
  %1 = pop.index_to_pointer %idx : !meta.pointer<!meta.scalar<f32>>
  // CHECK: pop.index_to_pointer %{{.*}} : !meta.pointer<!meta.simd<4, f32>>
  %2 = pop.index_to_pointer %idx : !meta.pointer<!meta.simd<4, f32>>
  // CHECK: pop.index_to_pointer %{{.*}} : !meta.pointer<?>
  %3 = pop.index_to_pointer %idx : !meta.pointer<?>
  kgen.return
}
