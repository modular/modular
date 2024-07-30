// RUN: kgen-opt -verify-parameters %s -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.generator @pointer_type<dt: dtype>(
kgen.generator @pointer_type<dt: dtype>
  // CHECK-SAME: %{{.*}}: !kgen.pointer<scalar<dt>>,
  (%arg0 : !kgen.pointer<scalar<dt>>,
  // CHECK-SAME: %{{.*}}: !kgen.pointer<scalar<ui8>>,
  %arg1: !kgen.pointer<scalar<ui8>>,
  // CHECK-SAME: %{{.*}}: !kgen.pointer<scalar<invalid>>) {
  %arg2: !kgen.pointer<scalar<invalid>>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @simd_type<dt: dtype, size>(
kgen.generator @simd_type<dt: dtype, size>
  // CHECK-SAME: %{{.*}}: !pop.simd<4, dt>,
  (%arg0 : !pop.simd<4,dt>,
  // CHECK-SAME: %{{.*}}: !pop.simd<mul(size, size), ui8>) {
   %arg1: !pop.simd<mul(size,size), ui8>) {

  kgen.return
}

// CHECK-LABEL: kgen.func @pop_neg
// CHECK-SAME: %[[ARG0:.*]]: !pop.scalar<f32>
// CHECK-SAME: %[[ARG1:.*]]: !pop.scalar<index>
kgen.func @pop_neg(%arg0: !pop.scalar<f32>,
                   %arg1: !pop.scalar<index>) -> (!pop.scalar<f32>,
                                                  !pop.scalar<index>) {
  // CHECK: pop.neg %[[ARG0]] : !pop.scalar<f32>
  %0 = pop.neg %arg0 : !pop.scalar<f32>
  // CHECK: pop.neg %[[ARG1]] : !pop.scalar<index>
  %1 = pop.neg %arg1 : !pop.scalar<index>
  kgen.return %0, %1 : !pop.scalar<f32>, !pop.scalar<index>
}

// CHECK-LABEL: kgen.func @pop_add
kgen.func @pop_add(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %arg0, %arg1 : !pop.scalar<f32>
  %c = pop.add %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %c : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.generator @pop_add2<DT: dtype>
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<DT>, %[[ARG1:.*]]: !pop.scalar<DT>) -> !pop.scalar<DT> {
kgen.generator @pop_add2<DT: dtype>(%a: !pop.scalar<DT>, %b: !pop.scalar<DT>) -> !pop.scalar<DT> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !pop.scalar<DT>
  %c = pop.add %a, %b : !pop.scalar<DT>
  kgen.return %c : !pop.scalar<DT>
}

// CHECK-LABEL: kgen.func @pop_add_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_add_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.add %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: kgen.func @pop_add_simd_index
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, index>, %[[ARG1:.*]]: !pop.simd<4, index>) -> !pop.simd<4, index> {
kgen.func @pop_add_simd_index(%arg0 : !pop.simd<4, index>, %arg1 : !pop.simd<4, index>) -> !pop.simd<4, index> {
  // CHECK-NEXT: %[[V0:.*]] = pop.add %[[ARG0]], %[[ARG1]] : !pop.simd<4, index>
  %0 = pop.add %arg0, %arg1 : !pop.simd<4, index>
  kgen.return %0 : !pop.simd<4, index>
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

// CHECK-LABEL: kgen.func @pop_sub_simd_index
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, index>, %[[ARG1:.*]]: !pop.simd<4, index>) -> !pop.simd<4, index> {
kgen.func @pop_sub_simd_index(%arg0 : !pop.simd<4, index>, %arg1 : !pop.simd<4, index>) -> !pop.simd<4, index> {
  // CHECK-NEXT: %[[V0:.*]] = pop.sub %[[ARG0]], %[[ARG1]] : !pop.simd<4, index>
  %0 = pop.sub %arg0, %arg1 : !pop.simd<4, index>
  kgen.return %0 : !pop.simd<4, index>
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

// CHECK-LABEL: kgen.func @pop_rem
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>, %[[ARG1:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @pop_rem(%arg0 : !pop.scalar<f32>, %arg1 : !pop.scalar<f32>) -> !pop.scalar<f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.rem %[[ARG0]], %[[ARG1]] : !pop.scalar<f32>
  %0 = pop.rem %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @pop_rem_simd
// CHECK-SAME: (%[[ARG0:.*]]: !pop.simd<4, f32>, %[[ARG1:.*]]: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
kgen.func @pop_rem_simd(%arg0 : !pop.simd<4, f32>, %arg1 : !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  // CHECK-NEXT: %[[V0:.*]] = pop.rem %[[ARG0]], %[[ARG1]] : !pop.simd<4, f32>
  %0 = pop.rem %arg0, %arg1 : !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// CHECK-LABEL: @pop_shifts
kgen.func @pop_shifts(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>,
                      %arg2: !pop.scalar<index>, %arg3: !pop.scalar<index>) {
  // CHECK: = pop.shl %{{.*}}, %{{.*}} : !pop.scalar<si32>
  %0 = pop.shl %arg0, %arg1 : !pop.scalar<si32>
  // CHECK: = pop.shr %{{.*}}, %{{.*}} : !pop.scalar<si32>
  %1 = pop.shr %arg0, %arg1 : !pop.scalar<si32>
  // CHECK: = pop.shr %{{.*}}, %{{.*}} : !pop.scalar<index>
  %2 = pop.shr %arg2, %arg3 : !pop.scalar<index>
  // CHECK: = pop.shl %{{.*}}, %{{.*}} : !pop.scalar<index>
  %3 = pop.shl %arg2, %arg3 : !pop.scalar<index>
  kgen.return
}

// CHECK-LABEL: @pop_shifts_simd
kgen.func @pop_shifts_simd(%arg0: !pop.simd<4, si32>, %arg1: !pop.simd<4, si32>,
                      %arg2: !pop.simd<4, index>, %arg3: !pop.simd<4, index>) {
  // CHECK: pop.shl %{{.*}}, %{{.*}} : !pop.simd<4, si32>
  %0 = pop.shl %arg0, %arg1 : !pop.simd<4, si32>
  // CHECK: pop.shr %{{.*}}, %{{.*}} : !pop.simd<4, si32>
  %1 = pop.shr %arg0, %arg1 : !pop.simd<4, si32>
  // CHECK: = pop.shr %{{.*}}, %{{.*}} : !pop.simd<4, index>
  %2 = pop.shr %arg2, %arg3 : !pop.simd<4, index>
  // CHECK: = pop.shl %{{.*}}, %{{.*}} : !pop.simd<4, index>
  %3 = pop.shl %arg2, %arg3 : !pop.simd<4, index>
  kgen.return
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

// CHECK-LABEL: @pop_and_bool
kgen.func @pop_and_bool(%arg0: !pop.scalar<bool>, %arg1: !pop.scalar<bool>,
                        %arg2: !pop.simd<4, bool>, %arg3: !pop.simd<4, bool>) {
  // CHECK: pop.and %{{.*}}, %{{.*}} :
  %0 = pop.and %arg0, %arg1 : !pop.scalar<bool>
  // CHECK: pop.and %{{.*}}, %{{.*}} :
  %1 = pop.and %arg2, %arg3 : !pop.simd<4, bool>
  kgen.return
}

// CHECK-LABEL: @pop_and
kgen.func @pop_and(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>,
                   %arg2: !pop.simd<4, si32>, %arg3: !pop.simd<4, si32>) {
  // CHECK: pop.and %{{.*}}, %{{.*}} :
  %0 = pop.and %arg0, %arg1 : !pop.scalar<si32>
  // CHECK: pop.and %{{.*}}, %{{.*}} :
  %1 = pop.and %arg2, %arg3 : !pop.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_and_index
kgen.func @pop_and_index(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>,
                       %arg2: !pop.simd<4, index>, %arg3: !pop.simd<4, index>) {
  // CHECK: pop.and %{{.*}}, %{{.*}} :
  %0 = pop.and %arg0, %arg1 : !pop.scalar<index>
  // CHECK: pop.and %{{.*}}, %{{.*}} :
  %1 = pop.and %arg2, %arg3 : !pop.simd<4, index>
  kgen.return
}

kgen.generator @pop_and_parametric<size, DT: dtype>(
                   %arg0: !pop.scalar<DT>, %arg1: !pop.scalar<DT>,
                   %arg2: !pop.simd<size, DT>, %arg3: !pop.simd<size, DT>) {
  // CHECK: pop.and
  %0 = pop.and %arg0, %arg1 : !pop.scalar<DT>
  // CHECK: pop.and
  %1 = pop.and %arg2, %arg3 : !pop.simd<size, DT>
  kgen.return
}

// CHECK-LABEL: @pop_or_bool
kgen.func @pop_or_bool(%arg0: !pop.scalar<bool>, %arg1: !pop.scalar<bool>,
                       %arg2: !pop.simd<4, bool>, %arg3: !pop.simd<4, bool>) {
  // CHECK: pop.or %{{.*}}, %{{.*}} :
  %0 = pop.or %arg0, %arg1 : !pop.scalar<bool>
  // CHECK: pop.or %{{.*}}, %{{.*}} :
  %1 = pop.or %arg2, %arg3 : !pop.simd<4, bool>
  kgen.return
}

// CHECK-LABEL: @pop_or
kgen.func @pop_or(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>,
                   %arg2: !pop.simd<4, si32>, %arg3: !pop.simd<4, si32>) {
  // CHECK: pop.or %{{.*}}, %{{.*}} :
  %0 = pop.or %arg0, %arg1 : !pop.scalar<si32>
  // CHECK: pop.or %{{.*}}, %{{.*}} :
  %1 = pop.or %arg2, %arg3 : !pop.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_or_index
kgen.func @pop_or_index(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>,
                       %arg2: !pop.simd<4, index>, %arg3: !pop.simd<4, index>) {
  // CHECK: pop.or %{{.*}}, %{{.*}} :
  %0 = pop.or %arg0, %arg1 : !pop.scalar<index>
  // CHECK: pop.or %{{.*}}, %{{.*}} :
  %1 = pop.or %arg2, %arg3 : !pop.simd<4, index>
  kgen.return
}

kgen.generator @pop_or_parametric<size, DT: dtype>(
                   %arg0: !pop.scalar<DT>, %arg1: !pop.scalar<DT>,
                   %arg2: !pop.simd<size, DT>, %arg3: !pop.simd<size, DT>) {
  // CHECK: pop.or
  %0 = pop.or %arg0, %arg1 : !pop.scalar<DT>
  // CHECK: pop.or
  %1 = pop.or %arg2, %arg3 : !pop.simd<size, DT>
  kgen.return
}

// CHECK-LABEL: @pop_xor_bool
kgen.func @pop_xor_bool(%arg0: !pop.scalar<bool>, %arg1: !pop.scalar<bool>,
                   %arg2: !pop.simd<4, bool>, %arg3: !pop.simd<4, bool>) {
  // CHECK: pop.xor %{{.*}}, %{{.*}} :
  %0 = pop.xor %arg0, %arg1 : !pop.scalar<bool>
  // CHECK: pop.xor %{{.*}}, %{{.*}} :
  %1 = pop.xor %arg2, %arg3 : !pop.simd<4, bool>
  kgen.return
}

// CHECK-LABEL: @pop_xor
kgen.func @pop_xor(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>,
                   %arg2: !pop.simd<4, si32>, %arg3: !pop.simd<4, si32>) {
  // CHECK: pop.xor %{{.*}}, %{{.*}} :
  %0 = pop.xor %arg0, %arg1 : !pop.scalar<si32>
  // CHECK: pop.xor %{{.*}}, %{{.*}} :
  %1 = pop.xor %arg2, %arg3 : !pop.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_xor_index
kgen.func @pop_xor_index(%arg0: !pop.scalar<index>, %arg1: !pop.scalar<index>,
                       %arg2: !pop.simd<4, index>, %arg3: !pop.simd<4, index>) {
  // CHECK: pop.xor %{{.*}}, %{{.*}} :
  %0 = pop.xor %arg0, %arg1 : !pop.scalar<index>
  // CHECK: pop.xor %{{.*}}, %{{.*}} :
  %1 = pop.xor %arg2, %arg3 : !pop.simd<4, index>
  kgen.return
}

kgen.generator @pop_xor_parametric<size, DT: dtype>(
                   %arg0: !pop.scalar<DT>, %arg1: !pop.scalar<DT>,
                   %arg2: !pop.simd<size, DT>, %arg3: !pop.simd<size, DT>) {
  // CHECK: pop.xor
  %0 = pop.xor %arg0, %arg1 : !pop.scalar<DT>
  // CHECK: pop.xor
  %1 = pop.xor %arg2, %arg3 : !pop.simd<size, DT>
  kgen.return
}

// CHECK-LABEL: @pop_select
kgen.func @pop_select(%arg0: i1, %arg1: !kgen.struct<(f32)>, %arg2: !kgen.struct<(f32)>) -> !kgen.struct<(f32)> {
  // CHECK: pop.select %arg0, %arg1, %arg2 : !kgen.struct<(f32)>
  %0 = pop.select %arg0, %arg1, %arg2 : !kgen.struct<(f32)>
  kgen.return %0 : !kgen.struct<(f32)>
}

// CHECK-LABEL: @pop_select_simd
kgen.func @pop_select_simd(
    %arg0: !pop.simd<4, bool>,
    %arg1: !pop.simd<4, si32>,
    %arg2: !pop.simd<4, si32>
  ) -> !pop.simd<4, si32> {
  // CHECK: pop.simd.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.simd.select %arg0, %arg1, %arg2 : !pop.simd<4, si32>
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
  // CHECK: %[[V5:.*]] = pop.bitcast %[[V0]] : !pop.simd<4, si32> to !pop.simd<128, bool>
  %5 = pop.bitcast %0 : !pop.simd<4, si32> to !pop.simd<128, bool>
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
kgen.generator @pointer_bitcast(%arg0: !kgen.pointer<scalar<f32>>, %arg1: !kgen.pointer<simd<4, f64>>) ->
   (!kgen.pointer<simd<4, si32>>, !kgen.pointer<scalar<f64>>) {
  // CHECK: %[[V0:.*]] = pop.pointer.bitcast %[[ARG0]] : !kgen.pointer<scalar<f32>> to !kgen.pointer<simd<4, si32>>
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<scalar<f32>> to !kgen.pointer<simd<4, si32>>
  // CHECK: %[[V1:.*]] = pop.pointer.bitcast %[[ARG1]] : !kgen.pointer<simd<4, f64>> to !kgen.pointer<scalar<f64>>
  %1 = pop.pointer.bitcast %arg1 : !kgen.pointer<simd<4, f64>> to !kgen.pointer<scalar<f64>>
  // CHECK: %{{.*}} = pop.pointer.bitcast %[[ARG0]] : !kgen.pointer<scalar<f32>> to !kgen.pointer<scalar<invalid>>
  %2 = pop.pointer.bitcast %arg0 : !kgen.pointer<scalar<f32>> to !kgen.pointer<scalar<invalid>>
  // CHECK: return %[[V0]], %[[V1]]
  kgen.return %0, %1 : !kgen.pointer<simd<4, si32>>, !kgen.pointer<scalar<f64>>
}

// CHECK-LABEL: @pointer_bitcast_funcptr
kgen.generator @pointer_bitcast_funcptr<T:type>(%arg0: !kgen.signature<() -> ()>) {
  // CHECK: pop.pointer.bitcast %arg0 : !kgen.signature<() -> ()> to !kgen.signature<(i32) -> i32>
  %0 = pop.pointer.bitcast %arg0 : !kgen.signature<() -> ()> to !kgen.signature<(i32) -> i32>
  // CHECK: pop.pointer.bitcast %arg0 : !kgen.signature<() -> ()> to !kgen.paramref<T>
  %1 = pop.pointer.bitcast %arg0 : !kgen.signature<() -> ()> to !kgen.paramref<T>
  kgen.return
}

// CHECK-LABEL: @pop_bitcast_paramref
// CHECK-SAME: %[[ARG:[a-z0-9]*]]:
kgen.generator @pop_bitcast_paramref<size1, dt1: dtype, size2, dt2: dtype>(%arg: !pop.simd<size1,dt1>) {
  // CHECK: pop.bitcast %[[ARG]] : !pop.simd<size1, dt1> to !pop.simd<size2, dt2>
  %0 = pop.bitcast %arg : !pop.simd<size1,dt1> to !pop.simd<size2,dt2>
  kgen.return
}

// CHECK-LABEL: @scalar_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @scalar_cast<DT: dtype>(%a: !pop.scalar<f32>) -> !pop.scalar<si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !pop.scalar<f32> to !pop.scalar<DT>
  %0 = pop.cast %a : !pop.scalar<f32> to !pop.scalar<DT>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !pop.scalar<DT> to !pop.scalar<f64>
  %1 = pop.cast %0 : !pop.scalar<DT> to !pop.scalar<f64>
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
kgen.generator @pop_simd_insertelement<size, DT: dtype>(
    %v0: !pop.scalar<DT>,
    %v1: !pop.scalar<f32>,
    %v2: !pop.scalar<si32>,
    %a: !pop.simd<size, DT>,
    %b: !pop.simd<size, f32>,
    %c: !pop.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK: %[[U:.*]] = pop.simd.insertelement %[[V0]], %[[A]][%[[IDX]]] : !pop.simd<size, DT>
  %u = pop.simd.insertelement %v0, %a[%idx] : !pop.simd<size, DT>
  // CHECK: %[[V:.*]] = pop.simd.insertelement %[[V1]], %[[B]][%[[IDX]]] : !pop.simd<size, f32>
  %v = pop.simd.insertelement %v1, %b[%idx] : !pop.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.insertelement %[[V2]], %[[C]][%[[IDX]]] : !pop.simd<4, si32>
  %w = pop.simd.insertelement %v2, %c[%idx] : !pop.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_simd_shuffle
kgen.generator @pop_simd_shuffle<size, mask: !pop.array<2,index>>(%a: !pop.simd<size, f32>, %b: !pop.simd<size, f32>) {
  // CHECK: pop.simd.shuffle <size, f32> %{{.*}}, %{{.*}} -> <2, f32> :array<2, index> [1, 2]
  %0 = pop.simd.shuffle <size, f32> %a, %b -> <2, f32> :array<2, index> [1, 2]
  // CHECK: pop.simd.shuffle <size, f32> %{{.*}}, %{{.*}} -> <4, f32> :array<4, index> [1, 2, 3, 4]
  %1 = pop.simd.shuffle <size, f32> %a, %b -> <4, f32> :array<4, index> [1, 2, 3, 4]
  kgen.return
}

// CHECK-LABEL: @pop_simd_splat
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_splat<size, DT: dtype>(%a: !pop.scalar<f32>, %b: !pop.scalar<DT>) -> (!pop.simd<4, f32>, !pop.simd<size, DT>) {
  // CHECK: %[[U:.*]] = pop.simd.splat %[[A]] : !pop.simd<4, f32>
  %u = pop.simd.splat %a : !pop.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.splat %[[B]] : !pop.simd<size, DT>
  %v = pop.simd.splat %b : !pop.simd<size, DT>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !pop.simd<4, f32>, !pop.simd<size, DT>
}

// CHECK-LABEL: @pop_load_store
kgen.generator @pop_load_store<DT: dtype>(%p0: !kgen.pointer<scalar<f32>>, %p1: !kgen.pointer<scalar<DT>>) {
  // CHECK: %[[V0:.*]] = pop.load %{{.*}} : !kgen.pointer<scalar<f32>>
  %0 = pop.load %p0 : !kgen.pointer<scalar<f32>>
  // CHECK: %[[V1:.*]] = pop.load %{{.*}} : !kgen.pointer<scalar<DT>>
  %1 = pop.load %p1 : !kgen.pointer<scalar<DT>>
  // CHECK: pop.store %[[V0]], %{{.*}} : !kgen.pointer<scalar<f32>>
  pop.store %0, %p0 : !kgen.pointer<scalar<f32>>
  // CHECK: pop.store %[[V1]], %{{.*}} : !kgen.pointer<scalar<DT>>
  pop.store %1, %p1 : !kgen.pointer<scalar<DT>>
  kgen.return
}

// CHECK-LABEL: @pop_load_store_alignment
kgen.generator @pop_load_store_alignment<DT: dtype>(%p0: !kgen.pointer<scalar<f32>>, %p1: !kgen.pointer<scalar<DT>>) {
  // CHECK: pop.load %{{.*}} align<42> : !kgen.pointer<scalar<f32>>
  %0 = pop.load %p0 align<42> : !kgen.pointer<scalar<f32>>
  // CHECK: pop.load %{{.*}} align<8> : !kgen.pointer<scalar<DT>>
  %1 = pop.load %p1 align<8> : !kgen.pointer<scalar<DT>>
  // CHECK: pop.store %{{.*}}, %{{.*}} align<4> : !kgen.pointer<scalar<f32>>
  pop.store %0, %p0 align<4> : !kgen.pointer<scalar<f32>>
  // CHECK: pop.store %{{.*}}, %{{.*}} align<89> : !kgen.pointer<scalar<DT>>
  pop.store %1, %p1 align<89> : !kgen.pointer<scalar<DT>>
  kgen.return
}

// CHECK-LABEL: @alignment_syntax
kgen.generator @alignment_syntax(%arg0: !kgen.pointer<index>) {
  // CHECK: align<#some.int>
  pop.load %arg0 align<#some.int> : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: @pop_load_alignment_generator
kgen.generator @pop_load_alignment_generator<alignment>(%ptr: !kgen.pointer<scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: pop.load %{{.*}} align<alignment> : !kgen.pointer<scalar<f32>>
  %0 = pop.load %ptr align<alignment> : !kgen.pointer<scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}

// CHECK-LABEL: @pop_offset
kgen.generator @pop_offset<type: dtype>(%p: !kgen.pointer<scalar<f32>>, %idx: index) {
  // pop.offset %{{.*}}[{{.*}}] : !kgen.pointer<scalar<f32>>
  %0 = pop.offset %p[%idx] : !kgen.pointer<scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @pop_generic_load_store
kgen.generator @pop_generic_load_store<ty: type, dt: dtype, size>(
    %p0: !kgen.pointer<ty>,
    %p1: !kgen.pointer<scalar<dt>>,
    %p2: !kgen.pointer<simd<size, dt>>)
  -> (
    !kgen.paramref<ty>,
    !pop.scalar<dt>,
    !pop.simd<size, dt>
  ) {
  // CHECK: pop.load %{{.*}} : !kgen.pointer<ty>
  // CHECK: pop.store %{{.*}} : !kgen.pointer<ty>
  %0 = pop.load %p0 : !kgen.pointer<ty>
  pop.store %0, %p0 : !kgen.pointer<ty>

  // CHECK: pop.load %{{.*}} : !kgen.pointer<scalar<dt>>
  // CHECK: pop.store %{{.*}} : !kgen.pointer<scalar<dt>>
  %1 = pop.load %p1 : !kgen.pointer<scalar<dt>>
  pop.store %1, %p1 : !kgen.pointer<scalar<dt>>

  // CHECK: pop.load %{{.*}} : !kgen.pointer<simd<size, dt>>
  // CHECK: pop.store %{{.*}} : !kgen.pointer<simd<size, dt>>
  %2 = pop.load %p2 : !kgen.pointer<simd<size, dt>>
  pop.store %2, %p2 : !kgen.pointer<simd<size, dt>>

  kgen.return %0, %1, %2 : !kgen.paramref<ty>, !pop.scalar<dt>, !pop.simd<size, dt>
}

// CHECK-LABEL: @pop_generic_offset
kgen.generator @pop_generic_offset<ty: type>(
    %p0: !kgen.pointer<ty>,
    %p1: !kgen.pointer<simd<4, f32>>,
    %i: index) {
  // CHECK: pop.offset %{{.*}} : !kgen.pointer<ty>
  %0 = pop.offset %p0[%i] : !kgen.pointer<ty>
  // CHECK: pop.offset %{{.*}} : !kgen.pointer<simd<4, f32>>
  %1 = pop.offset %p1[%i] : !kgen.pointer<simd<4, f32>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @parametricAdd
// CHECK-SAME: %[[ARG0:.*]]: !pop.simd<size, dt>, %[[ARG1:.*]]: !pop.simd<size, dt>
kgen.generator @parametricAdd<size, dt: dtype>
  (%arg0: !pop.simd<size, dt>, %arg1: !pop.simd<size, dt>) -> !pop.simd<size, dt> {

  // Fully parametric operations are ok!
  // CHECK: %{{.*}} = pop.add %[[ARG0]], %[[ARG1]] : !pop.simd<size, dt>
  %0 = pop.add %arg0, %arg1 : !pop.simd<size,dt>
  kgen.return %0 : !pop.simd<size,dt>
}

// CHECK-LABEL: @stack_allocation
kgen.generator @stack_allocation<size, ty: type, address_space_val>() {
  // CHECK: pop.stack_allocation size x ty
  %0 = pop.stack_allocation size x ty
  // CHECK: pop.stack_allocation 16 x simd<4, f32>
  %1 = pop.stack_allocation 16 x !pop.simd<4, f32>
  // CHECK: pop.stack_allocation 16 x simd<4, f32> align 8
  %2 = pop.stack_allocation 16 x !pop.simd<4, f32> align 8
  // CHECK: pop.stack_allocation 16 x simd<4, f32> align size
  %3 = pop.stack_allocation 16 x !pop.simd<4, f32> align size
  // CHECK: pop.stack_allocation 16 x simd<4, f32> address_space 5
  %4 = pop.stack_allocation 16 x !pop.simd<4, f32> address_space 5
  // CHECK: pop.stack_allocation 16 x simd<4, f32> address_space 5 align 8
  %5 = pop.stack_allocation 16 x !pop.simd<4, f32> address_space 5 align 8
  // CHECK: pop.stack_allocation 16 x simd<4, f32> address_space address_space_val
  %6 = pop.stack_allocation 16 x !pop.simd<4, f32> address_space address_space_val
  kgen.return
}

// CHECK-LABEL: @external_call
kgen.generator @external_call<ty: type, dt: dtype>(%a: !kgen.paramref<ty>, %b: !pop.scalar<dt>) {
  // CHECK: pop.external_call @foo(%{{.*}}, %{{.*}})
  %0 = pop.external_call @foo(%a, %b) : (!kgen.paramref<ty>, !pop.scalar<dt>) -> !pop.simd<4, f32>
  // CHECK: pop.external_call @bar(%arg0, %arg1)
  // CHECK-SAME: (!kgen.paramref<ty>) -> ()
  // CHECK-SAME: attributes {funcAttrs = ["noinline", ["alignstack", "16"]]}
  pop.external_call @bar(%a, %b) (!kgen.paramref<ty>) -> ()
    attributes {funcAttrs = ["noinline", ["alignstack", "16"]]}
    : (!kgen.paramref<ty>, !pop.scalar<dt>) -> ()
  kgen.return
}

// CHECK-LABEL: @global_constant
kgen.generator @global_constant() {
  // CHECK: pop.global_constant: i32 = <5>
  pop.global_constant: i32 = <5>
  // CHECK: pop.global_constant: !M.array<4xui32> = <#M.dense_array<0, 1, 2, 3>>
  pop.global_constant: !M.array<4xui32> = <#M.dense_array<0, 1, 2, 3>>
  kgen.return
}

// CHECK-LABEL: @global_alloc
kgen.generator @global_alloc() {
  // CHECK-NEXT: pop.global_alloc 2 x scalar<si32> address_space 3 align 32
  %0 = pop.global_alloc 2 x !pop.scalar<si32> address_space 3 align 32
  kgen.return
}

// CHECK-LABEL: @global_constant_aligned
kgen.generator @global_constant_aligned() {
  // CHECK: pop.global_constant: i32 = <5> align 16
  pop.global_constant: i32 = <5> align 16
  // CHECK: pop.global_constant: !M.array<4xui32> = <#M.dense_array<0, 1, 2, 3>>  align 64
  pop.global_constant: !M.array<4xui32> = <#M.dense_array<0, 1, 2, 3>> align 64
  kgen.return
}

// CHECK-LABEL: @pointer_to_index
kgen.generator @pointer_to_index<ty: type>(%a: !kgen.pointer<ty>, %b: !kgen.pointer<scalar<f32>>) {
  // CHECK: pop.pointer_to_index %{{.*}} : <ty>
  %0 = pop.pointer_to_index %a : !kgen.pointer<ty>
  // CHECK: pop.pointer_to_index %{{.*}} : <scalar<f32>>
  %1 = pop.pointer_to_index %b : !kgen.pointer<scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @struct
kgen.generator @struct<ty: type, dt: dtype>(
  // CHECK-SAME: %[[A:.*]]: !kgen.paramref
  %a: !kgen.paramref<ty>,
  // CHECK-SAME: %[[B:.*]]: !pop.scalar<
  %b: !pop.scalar<dt>
) -> (!kgen.paramref<ty>, !pop.scalar<dt>, !kgen.pointer<ty>) {
  // CHECK: %[[S0:.*]] = kgen.struct.create(%[[A]], %[[B]]) : !kgen.struct<(ty, scalar<dt>)>
  %0 = kgen.struct.create(%a, %b) : !kgen.struct<(ty, scalar<dt>)>
  // CHECK: %[[V0:.*]] = kgen.struct.extract %[[S0]][0] : !kgen.struct<(ty, scalar<dt>)>
  %1 = kgen.struct.extract %0[0] : !kgen.struct<(ty, scalar<dt>)>
  // CHECK: %[[V1:.*]] = kgen.struct.extract %[[S0]][1] : !kgen.struct<(ty, scalar<dt>)>
  %2 = kgen.struct.extract %0[1] : !kgen.struct<(ty, scalar<dt>)>
  // CHECK: kgen.struct.replace %{{.*}}, %[[S0]][0] : !kgen.struct<(ty, scalar<dt>)>
  %3 = kgen.struct.replace %1, %0[0] : !kgen.struct<(ty, scalar<dt>)>
  // CHECK: kgen.struct.replace %{{.*}}, %{{.*}}[1] : !kgen.struct<(ty, scalar<dt>)>
  %4 = kgen.struct.replace %2, %3[1] : !kgen.struct<(ty, scalar<dt>)>

  // CHECK: %[[STRUCT_PTR:.*]] = pop.stack_allocation
  %struct = pop.stack_allocation 1 x !kgen.struct<(i32, ty)>
  // CHECK: %[[EL_PTR:.*]] = kgen.struct.gep %[[STRUCT_PTR]][1] : <struct<(i32, ty)>>
  %el = kgen.struct.gep %struct[1] : <struct<(i32, ty)>>

  // CHECK: return %[[V0]], %[[V1]], %[[EL_PTR]] : !kgen.paramref<ty>, !pop.scalar<dt>, !kgen.pointer<ty>
  kgen.return %1, %2, %el : !kgen.paramref<ty>, !pop.scalar<dt>, !kgen.pointer<ty>
}

// CHECK-LABEL: @empty_struct_syntax
kgen.generator @empty_struct_syntax() -> !kgen.struct<()> {
  // CHECK-NEXT: kgen.struct.create() : !kgen.struct<()>
  %0 = kgen.struct.create() : !kgen.struct<()>
  kgen.return %0 : !kgen.struct<()>
}

// CHECK-LABEL: @pointer_types
kgen.generator @pointer_types<dt: dtype>(
  // CHECK-SAME: %{{.*}}: !kgen.pointer<scalar<dt>>, %{{.*}}: !kgen.pointer<scalar<f32>>, %{{.*}}: !kgen.pointer<scalar<invalid>>
  %arg0: !kgen.pointer<scalar<dt>>, %arg1: !kgen.pointer<scalar<f32>>, %arg2: !kgen.pointer<scalar<invalid>>) {
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
kgen.func @cast_from_builtin_vector(%arg0: vector<2xf32>) -> !pop.simd<2, f32> {
  // CHECK: %[[V0:.*]] = pop.cast_from_builtin %[[ARG]] : vector<2xf32> to !pop.simd<2, f32>
  %0 = pop.cast_from_builtin %arg0 : vector<2xf32> to !pop.simd<2, f32>
  // CHECK: kgen.return %[[V0:.*]] : !pop.simd<2, f32>
  kgen.return %0 : !pop.simd<2, f32>
}

// CHECK-LABEL: @array_ops
kgen.generator @array_ops<idx, N, T: type, dtype: dtype>(%arg0: !kgen.paramref<T>)
    -> (!pop.array<2, T>, !kgen.pointer<T>) {
  // CHECK: pop.array.create [%arg0, %arg0] : !pop.array<2, T>
  %0 = pop.array.create [%arg0, %arg0] : !pop.array<2, T>
  // CHECK: pop.array.get %0[1] : !pop.array<2, T>
  %1 = pop.array.get %0[1] : !pop.array<2, T>
  // CHECK: pop.array.replace %{{.*}}, %0[0] : !pop.array<2, T>
  %2 = pop.array.replace %1, %0[0] : !pop.array<2, T>

  // CHECK: %[[ARR_PTR:.*]] = pop.stack_allocation 1 x array<4, T>
  %5 = pop.stack_allocation 1 x !pop.array<4, T>
  // CHECK: %[[IDX:.*]] = index.constant
  %6 = index.constant 2
  // CHECK: pop.array.gep %[[ARR_PTR]][%[[IDX]]] : <array<4, T>>
  %7 = pop.array.gep %5[%6] : <array<4, T>>

  // CHECK: pop.array.get %{{.*}}[idx]
  %8 = pop.array.get %0[idx] : !pop.array<2, T>
  // CHECK: pop.array.replace %arg0, %{{.*}}[idx]
  %9 = pop.array.replace %arg0, %0[idx] : !pop.array<2, T>

  kgen.return %2, %7 : !pop.array<2, T>, !kgen.pointer<T>
}

// CHECK-LABEL: kgen.generator @pack
kgen.generator @pack<Ts: variadic<!kgen.type>, T: type, I: index>(
  %arg0: !kgen.pack<Ts>,
  %arg1: !kgen.pack<[i32, T]>,
  %arg2: f32,
  %arg3: i8
) -> i32 {
  // CHECK: kgen.pack.size %arg0 : <Ts>
  %0 = kgen.pack.size %arg0 : <Ts>
  // CHECK: kgen.pack.extract %arg0[3] : <Ts>
  %1 = kgen.pack.extract %arg0[3] : <Ts>

  // CHECK: kgen.pack.size %arg1 : <[i32, T]>
  %2 = kgen.pack.size %arg1 : <[i32, T]>
  // CHECK: kgen.pack.extract %arg1[0] : <[i32, T]>
  %3 = kgen.pack.extract %arg1[0] : <[i32, T]>
  // CHECK: kgen.pack.extract %arg1[1] : <[i32, T]>
  %4 = kgen.pack.extract %arg1[1] : <[i32, T]>
  // CHECK: kgen.pack.extract %arg1[add(I, 1)] : <[i32, T]>
  %5 = kgen.pack.extract %arg1[add(I, 1)] : <[i32, T]>

  // CHECK: %[[PACK:.*]] = kgen.pack.create(%arg2, %arg2, %arg3) : !kgen.pack<[f32, f32, i8]>
  %6 = kgen.pack.create(%arg2, %arg2, %arg3) : !kgen.pack<[f32, f32, i8]>
  // CHECK: kgen.pack.size %[[PACK]] : <[f32, f32, i8]>
  %7 = kgen.pack.size %6 : <[f32, f32, i8]>
  // CHECK: kgen.pack.create() : !kgen.pack<[]>
  %8 = kgen.pack.create() : !kgen.pack<[]>

  kgen.return %3 : i32
}


// CHECK-LABEL: kgen.generator @pack
kgen.generator @pack_ptr<Ts: variadic<!kgen.type>, T: type, I: index>(
  %arg0: !kgen.pointer<!kgen.pack<Ts>>,
  %arg1: !kgen.pointer<!kgen.pack<[i32, T]>>
)  {
  // CHECK: kgen.pack.gep %arg0[3] : <!kgen.pack<Ts>>
  %1 = kgen.pack.gep %arg0[3] : <!kgen.pack<Ts>>

  // CHECK: kgen.pack.gep %arg1[0] : <!kgen.pack<[i32, T]>>
  %2 = kgen.pack.gep %arg1[0] : <!kgen.pack<[i32, T]>>
  // CHECK: kgen.pack.gep %arg1[1] : <!kgen.pack<[i32, T]>>
  %3 = kgen.pack.gep %arg1[1] : <!kgen.pack<[i32, T]>>
  // CHECK: kgen.pack.gep %arg1[add(I, 1)] : <!kgen.pack<[i32, T]>>
  %4 = kgen.pack.gep %arg1[add(I, 1)] : <!kgen.pack<[i32, T]>>

  kgen.return
}


// CHECK-LABEL: @parametric_pack
kgen.generator @parametric_pack<N, T: type>(%arg0: !pop.simd<N, bool>, %arg1: !kgen.paramref<T>) {
  // CHECK-NEXT: kgen.pack.create(%arg0, %arg1) : !kgen.pack<[simd<N, bool>, T]>
  %0 = kgen.pack.create(%arg0, %arg1) : !kgen.pack<[!pop.simd<N, bool>, T]>
  kgen.return
}

// CHECK-LABEL: @call_intrinsic
kgen.generator @call_intrinsic<intrin: string>(%arg0: !pop.scalar<f32>) {
  // CHECK-NEXT: %{{.*}} = pop.call_llvm_intrinsic "llvm.round", (%arg0) : (!pop.scalar<f32>) -> !pop.scalar<f32>
  %0 = pop.call_llvm_intrinsic "llvm.round", (%arg0) : (!pop.scalar<f32>) -> !pop.scalar<f32>
  // CHECK-NEXT: pop.call_llvm_intrinsic intrin, ()
  pop.call_llvm_intrinsic intrin, () : () -> ()
  kgen.return
}

// CHECK-LABEL: @inline_asm
kgen.generator @inline_asm<ty: type, dt: dtype>(
    %arg0: !pop.scalar<si32>,
    %arg1: !pop.scalar<index>,
    %arg2: !kgen.paramref<ty>,
    %arg3: !pop.scalar<dt>) {
  // CHECK: pop.inline_asm "bswap $0", "=r,r", (%arg0) : (!pop.scalar<si32>) -> i8
  %0 = pop.inline_asm "bswap $0", "=r,r", (%arg0) : (!pop.scalar<si32>) -> i8
  // CHECK: pop.inline_asm "something", "anotherthing", (%arg0, %arg1) :
  // CHECK: (!pop.scalar<si32>, !pop.scalar<index>) -> i8
  %1 = pop.inline_asm "something", "anotherthing", (%arg0, %arg1) :
    (!pop.scalar<si32>, !pop.scalar<index>) -> i8
  // CHECK: pop.inline_asm side_effecting "something", "anotherthing", (%arg0, %arg1) :
  // CHECK: (!pop.scalar<si32>, !pop.scalar<index>) -> i8
  %2 = pop.inline_asm side_effecting "something", "anotherthing", (%arg0, %arg1) :
    (!pop.scalar<si32>, !pop.scalar<index>) -> i8
  // CHECK: pop.inline_asm stack_aligned "something", "anotherthing", (%arg0, %arg1) :
  // CHECK: (!pop.scalar<si32>, !pop.scalar<index>) -> i8
  %3 = pop.inline_asm stack_aligned "something", "anotherthing", (%arg0, %arg1) :
    (!pop.scalar<si32>, !pop.scalar<index>) -> i8
  // CHECK: pop.inline_asm "foo", "=r,=r,r", (%arg0) : (!pop.scalar<si32>) ->
  // CHECK: !kgen.struct<(ty, scalar<dt>)>
  %4 = pop.inline_asm "foo", "=r,=r,r", (%arg0) : (!pop.scalar<si32>) ->
    !kgen.struct<(ty, scalar<dt>)>
  // CHECK: pop.inline_asm "bar $0", "=r,r", (%arg2) : (!kgen.paramref<ty>) -> i8
  %5 = pop.inline_asm "bar $0", "=r,r", (%arg2) : (!kgen.paramref<ty>) -> i8
  // CHECK: pop.inline_asm "bar $0", "=r,r", (%arg3) : (!pop.scalar<dt>) -> i8
  %6 = pop.inline_asm "bar $0", "=r,r", (%arg3) : (!pop.scalar<dt>) -> i8
  // CHECK: [[RET:%.*]]:2 = pop.inline_asm "bar $0", "=r,=r,r", (%arg0) : (!pop.scalar<si32>) -> (i8, i8)
  %7:2 = pop.inline_asm "bar $0", "=r,=r,r", (%arg0) : (!pop.scalar<si32>) -> (i8, i8)
  kgen.return
}

// CHECK-LABEL: kgen.generator @variadics
kgen.generator @variadics<ty: type>(
    %arg0: !pop.scalar<f32>,
    %arg1: !pop.scalar<f32>,
    %arg2: !kgen.struct<()>,
    %arg3: !kgen.struct<()>,
    %arg4: !kgen.paramref<ty>) {
  // CHECK: %[[V0:.*]] = pop.variadic.create [%arg0, %arg1] : !kgen.variadic<scalar<f32>>
  %v0 = pop.variadic.create [%arg0, %arg1] : !kgen.variadic<!pop.scalar<f32>>
  // CHECK: pop.variadic.size %[[V0]] : !kgen.variadic<scalar<f32>>
  %s0 = pop.variadic.size %v0 : !kgen.variadic<!pop.scalar<f32>>
  // CHECK: pop.variadic.get %[[V0]][%{{[a-zA-Z0-9]+}}] : !kgen.variadic<scalar<f32>>
  %i0 = index.constant 2
  %g0 = pop.variadic.get %v0[%i0] : !kgen.variadic<!pop.scalar<f32>>

  // CHECK: %[[V1:.*]] = pop.variadic.create [] : !kgen.variadic<scalar<f32>>
  %v1 = pop.variadic.create [] : !kgen.variadic<!pop.scalar<f32>>
  // CHECK: pop.variadic.size %[[V1]]
  %s1 = pop.variadic.size %v1 : !kgen.variadic<!pop.scalar<f32>>

  // CHECK: %[[V2:.*]] = pop.variadic.create [%arg2, %arg3] : !kgen.variadic<struct<()>>
  %v2 = pop.variadic.create [%arg2, %arg3] : !kgen.variadic<!kgen.struct<()>>
  // CHECK: pop.variadic.size %[[V2]]
  %s2 = pop.variadic.size %v2 : !kgen.variadic<!kgen.struct<()>>

  // CHECK: %[[V3:.*]] = pop.variadic.create [%arg4] : !kgen.variadic<ty>
  %v3 = pop.variadic.create [%arg4] : !kgen.variadic<ty>
  // CHECK: pop.variadic.size %[[V3]]
  %s3 = pop.variadic.size %v3 : !kgen.variadic<ty>

 // CHECK: %[[V3:.*]] = pop.variadic.splat 1, %arg4 : !kgen.variadic<ty>
  %v4 = pop.variadic.splat 1, %arg4 : !kgen.variadic<ty>

  kgen.return
}

// CHECK-LABEL: kgen.func @variadic_argument
kgen.func @variadic_argument(%arg0: !kgen.variadic<f32>) {
  // CHECK: pop.variadic.size %arg0
  %0 = pop.variadic.size %arg0 : !kgen.variadic<f32>
  kgen.return
}

// CHECK-LABEL: @usesAGlobal
kgen.func @usesAGlobal() {
  %zero = index.constant 0
  // CHECK: pop.compiler.global_load "aGlobal" : index
  %0 = pop.compiler.global_load "aGlobal" : index
  // CHECK: pop.compiler.global_store "aGlobal", %idx0 : index
  pop.compiler.global_store "aGlobal", %zero : index
  kgen.return
}

// CHECK-LABEL: kgen.func @atomic_cmpxchg
// CHECK-SAME: %[[PTR:.*]]: !kgen.pointer<scalar<index>>,
// CHECK-SAME: %[[CMP:.*]]: !pop.scalar<index>,
// CHECK-SAME: %[[NEW:.*]]: !pop.scalar<index>
kgen.func @atomic_cmpxchg(%ptr: !kgen.pointer<scalar<index>>,
                          %cmp: !pop.scalar<index>,
                          %new: !pop.scalar<index>) {
  // CHECK: pop.atomic.cmpxchg %[[PTR]], %[[CMP]], %[[NEW]] monotonic monotonic
  %0 = pop.atomic.cmpxchg %ptr, %cmp, %new monotonic monotonic :
                    !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.cmpxchg %[[PTR]], %[[CMP]], %[[NEW]] seq_cst acq_rel
  %1 = pop.atomic.cmpxchg %ptr, %cmp, %new seq_cst acq_rel :
                    !kgen.pointer<scalar<index>>
  kgen.return
}

// CHECK-LABEL: kgen.func @atomic_rmw
// CHECK-SAME: %[[PTR:.*]]: !kgen.pointer<scalar<index>>,
// CHECK-SAME: %[[VAL:.*]]: !pop.scalar<index>
kgen.func @atomic_rmw(%ptr: !kgen.pointer<scalar<index>>,
                      %val: !pop.scalar<index>) {
  // CHECK: pop.atomic.rmw add(%[[PTR]], %[[VAL]]) monotonic
  %0 = pop.atomic.rmw add(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw sub(%[[PTR]], %[[VAL]]) monotonic
  %1 = pop.atomic.rmw sub(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw xor(%[[PTR]], %[[VAL]]) monotonic
  %2 = pop.atomic.rmw xor(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw min(%[[PTR]], %[[VAL]]) monotonic
  %3 = pop.atomic.rmw min(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  // CHECK: pop.atomic.rmw max(%[[PTR]], %[[VAL]]) monotonic
  %4 = pop.atomic.rmw max(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
  kgen.return
}

// CHECK-LABEL: kgen.func @string_ops(%arg0: !kgen.string) -> index
kgen.func @string_ops(%a: !kgen.string) ->  index {
  // CHECK: pop.string.address %arg0
  %0 = pop.string.address %a
  // CHECK: pop.string.size %arg0
  %1 = pop.string.size %a
  // CHECK: pop.string.concat %arg0, %arg0
  %2 = pop.string.concat %a, %a
  kgen.return %1: index
}

// CHECK-LABEL: kgen.generator @dtype_utils
kgen.generator @dtype_utils<DT: dtype>(%arg0: !kgen.dtype) {
  // CHECK: %[[V0:.*]] = pop.dtype.to_ui8 %arg0
  %v0 = pop.dtype.to_ui8 %arg0
  // CHECK: pop.dtype.from_ui8 %[[V0]]
  %x0 = pop.dtype.from_ui8 %v0

  // CHECK: %[[PARAM:.*]] = kgen.param.constant
  %t0 = kgen.param.constant : dtype = <DT>

  // CHECK: %[[V1:.*]] = pop.dtype.to_ui8 %[[PARAM]]
  %v1 = pop.dtype.to_ui8 %t0
  // CHECK: pop.dtype.from_ui8 %[[V1]]
  %x1 = pop.dtype.from_ui8 %v1

  kgen.return
}

kgen.func @global_var_ctor() {
  kgen.return
}

// CHECK: kgen.global @global_var : i32 [@global_var_ctor, @global_var_ctor](2)
kgen.global @global_var : i32 [@global_var_ctor, @global_var_ctor](2)

// CHECK-LABEL: @global_address
kgen.func @global_address() -> !kgen.pointer<i32> {
  // CHECK-NEXT: kgen.global.address @global_var : <i32>
  %0 = kgen.global.address @global_var : <i32>
  kgen.return %0 : !kgen.pointer<i32>
}

// CHECK-LABEL: @aligned_alloc
kgen.func @aligned_alloc(%arg0: index, %arg1: index) {
  // CHECK-NEXT: %0 = pop.aligned_alloc %arg0, %arg1 : <index>
  %0 = pop.aligned_alloc %arg0, %arg1 : <index>
  // CHECK-NEXT: pop.aligned_free %0 : <index>
  pop.aligned_free %0 : <index>
  kgen.return
}

// CHECK-LABEL: @fence
kgen.func @fence() {
  // CHECK: pop.fence acquire
  pop.fence acquire
  // CHECK: pop.fence syncscope("agent") seq_cst
  pop.fence syncscope("agent") seq_cst
  // CHECK: pop.fence syncscope("singlethread") acq_rel
  pop.fence syncscope("singlethread") acq_rel
  kgen.return
}

// CHECK-LABEL: @stack_lifetime
kgen.func @stack_lifetime() {
  %0 = pop.stack_allocation 1 x index marked
  %1 = pop.stack_allocation 1 x index marked
  // CHECK: pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  // CHECK: pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<index>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: @variant_bitcast
kgen.generator @variant_bitcast<idx, Ts: variadic<type>>(%arg0: !kgen.pointer<variant<i32, i64>>, %arg1: !kgen.pointer<variant<[Ts]>>) -> (!kgen.pointer<i64>, !kgen.pointer<i32>) {
  // CHECK-NEXT: pop.variant.bitcast %arg0, <1> : <variant<i32, i64>> as <i64>
  %0 = pop.variant.bitcast %arg0, <1> : <variant<i32, i64>> as <i64>
  // CHECK-NEXT: pop.variant.bitcast %arg1, <idx> : <variant<[Ts]>> as <i32>
  %1 = pop.variant.bitcast %arg1, <idx> : <variant<[Ts]>> as <i32>
  kgen.return %0, %1 : !kgen.pointer<i64>, !kgen.pointer<i32>
}
