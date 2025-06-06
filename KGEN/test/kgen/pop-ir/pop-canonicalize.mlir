// RUN: kgen-opt -split-input-file %s -canonicalize | FileCheck %s

// CHECK-LABEL: @neg
kgen.func @neg() -> (!pop.simd<2, si8>, !pop.simd<2, f32>) {
  // CHECK-DAG: <1, -1>
  // CHECK-DAG: <"1.25", "-1.25">
  %0 = kgen.param.constant: simd<2, si8> = <<-1, 1>>
  %1 = kgen.param.constant: simd<2, f32> = <<"-1.25", "1.25">>
  %2 = pop.neg %0 : !pop.simd<2, si8>
  %3 = pop.neg %1 : !pop.simd<2, f32>
  kgen.return %2, %3 : !pop.simd<2, si8>, !pop.simd<2, f32>
}

// CHECK-LABEL: @add
kgen.func @add() -> (!pop.scalar<si8>, !pop.scalar<f32>) {
  // CHECK-DAG: <4>
  // CHECK-DAG: <"-2.5">
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = kgen.param.constant: scalar<f32> = <<"-1.25">>
  %2 = pop.add %0, %0 : !pop.scalar<si8>
  %3 = pop.add %1, %1 : !pop.scalar<f32>
  kgen.return %2, %3 : !pop.scalar<si8>, !pop.scalar<f32>
}

// CHECK-LABEL: @add_zero
kgen.func @add_zero(%arg0: !pop.simd<2, si8>,
                 %arg1: !pop.scalar<f32>) -> (!pop.simd<2, si8>,
                                              !pop.simd<2, si8>,
                                              !pop.simd<2, si8>,
                                              !pop.scalar<f32>) {
  %0 = kgen.param.constant: simd<2, si8> = <<0, 0>>
  // CHECK: %[[ZERO_ONE:.*]] = kgen.param.constant: simd<2, si8> = <<0, 1>>
  %1 = kgen.param.constant: simd<2, si8> = <<0, 1>>
  // CHECK: %[[FP_ZERO:.*]] = kgen.param.constant: scalar<f32> = <"0">
  %2 = kgen.param.constant: scalar<f32> = <<"0.0">>

  // x+0 -> x
  %3 = pop.add %arg0, %0 : !pop.simd<2, si8>
  // 0+x -> x
  %4 = pop.add %0, %arg0 : !pop.simd<2, si8>

  // negative case: %1 is not all zeros
  // CHECK: %[[R1:.*]] = pop.add %arg0, %[[ZERO_ONE]] : !pop.simd<2, si8>
  %5 = pop.add %arg0, %1 : !pop.simd<2, si8>

  // negative case: non-integer type
  // CHECK: %[[R2:.*]] = pop.add %arg1, %[[FP_ZERO]] : !pop.scalar<f32>
  %6 = pop.add %arg1, %2 : !pop.scalar<f32>

  // First two outputs are reduced to %arg0, remaining two are intact
  // CHECK: return %arg0, %arg0, %[[R1]], %[[R2]]
  kgen.return %3, %4, %5, %6 : !pop.simd<2, si8>, !pop.simd<2, si8>, !pop.simd<2, si8>, !pop.scalar<f32>
}

// CHECK-LABEL: @sub
kgen.func @sub() -> (!pop.scalar<si8>, !pop.scalar<f32>) {
  // CHECK-DAG: <-2>
  // CHECK-DAG: <"-1.25">
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = kgen.param.constant: scalar<si8> = <<4>>
  %2 = kgen.param.constant: scalar<f32> = <<"1.25">>
  %3 = kgen.param.constant: scalar<f32> = <<"2.5">>
  %4 = pop.sub %0, %1 : !pop.scalar<si8>
  %5 = pop.sub %2, %3 : !pop.scalar<f32>
  kgen.return %4, %5 : !pop.scalar<si8>, !pop.scalar<f32>
}

// CHECK-LABEL: @sub_zero
kgen.func @sub_zero(%arg0: !pop.simd<2, si8>,
                 %arg1: !pop.scalar<f32>) -> (!pop.simd<2, si8>,
                                              !pop.simd<2, si8>,
                                              !pop.simd<2, si8>,
                                              !pop.scalar<f32>) {
  // CHECK-DAG: %[[ZERO_ZERO:.*]] = kgen.param.constant: simd<2, si8> = <0>
  %0 = kgen.param.constant: simd<2, si8> = <<0, 0>>
  // CHECK-DAG: %[[ZERO_ONE:.*]] = kgen.param.constant: simd<2, si8> = <<0, 1>>
  %1 = kgen.param.constant: simd<2, si8> = <<0, 1>>
  // CHECK-DAG: %[[FP_ZERO:.*]] = kgen.param.constant: scalar<f32> = <"0">
  %2 = kgen.param.constant: scalar<f32> = <<"0.0">>

  // x-0 -> x
  %3 = pop.sub %arg0, %0 : !pop.simd<2, si8>

  // negative case: 0-x
  // CHECK: %[[R0:.*]] = pop.sub %[[ZERO_ZERO]], %arg0 : !pop.simd<2, si8>
  %4 = pop.sub %0, %arg0 : !pop.simd<2, si8>

  // negative case: vector value %1 is not all zeros
  // CHECK: %[[R1:.*]] = pop.sub %arg0, %[[ZERO_ONE]] : !pop.simd<2, si8>
  %5 = pop.sub %arg0, %1 : !pop.simd<2, si8>

  // negative case: non-integer type
  // CHECK: %[[R2:.*]] = pop.sub %arg1, %[[FP_ZERO]] : !pop.scalar<f32>
  %6 = pop.sub %arg1, %2 : !pop.scalar<f32>

  // First output is reduced to %arg0, remaining ones are intact
  // CHECK: return %arg0, %[[R0]], %[[R1]], %[[R2]]
  kgen.return %3, %4, %5, %6 : !pop.simd<2, si8>, !pop.simd<2, si8>, !pop.simd<2, si8>, !pop.scalar<f32>
}


// CHECK-LABEL: @mul
kgen.func @mul() -> (!pop.scalar<si8>, !pop.scalar<f32>) {
  // CHECK-DAG: <4>
  // CHECK-DAG: <"6.25">
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = kgen.param.constant: scalar<f32> = <<"2.5">>
  %2 = pop.mul %0, %0 : !pop.scalar<si8>
  %3 = pop.mul %1, %1 : !pop.scalar<f32>
  kgen.return %2, %3 : !pop.scalar<si8>, !pop.scalar<f32>
}

// CHECK-LABEL: @mul_zero_one
kgen.func @mul_zero_one(
    %si32: !pop.scalar<si32>,
    %vsi32 : !pop.simd<2, si32>) -> (!pop.scalar<si32>, !pop.scalar<si32>,
                                     !pop.scalar<si32>, !pop.scalar<si32>,
                                     !pop.simd<2, si32>, !pop.simd<2, si32>,
                                     !pop.simd<2, si32>, !pop.simd<2, si32>) {
  %zero = kgen.param.constant: scalar<si32> = <<0>>
  %one = kgen.param.constant: scalar<si32> = <<1>>
  %0 = pop.mul %si32, %zero : !pop.scalar<si32>
  %1 = pop.mul %zero, %si32 : !pop.scalar<si32>

  %2 = pop.mul %si32, %one : !pop.scalar<si32>
  %3 = pop.mul %one, %si32 : !pop.scalar<si32>

  %vzero = kgen.param.constant: simd<2, si32> = <<0, 0>>
  %vone = kgen.param.constant: simd<2, si32> = <<1, 1>>

  %4 = pop.mul %vsi32, %vzero : !pop.simd<2, si32>
  %5 = pop.mul %vzero, %vsi32 : !pop.simd<2, si32>

  %6 = pop.mul %vsi32, %vone : !pop.simd<2, si32>
  %7 = pop.mul %vone, %vsi32 : !pop.simd<2, si32>

  // CHECK-DAG: %[[SIMD_ZERO:.*]] = kgen.param.constant: simd<2, si32> = <0>
  // CHECK-DAG: %[[ZERO:.*]] = kgen.param.constant: scalar<si32> = <0>
  // CHECK: kgen.return %[[ZERO]], %[[ZERO]], %arg0, %arg0, %[[SIMD_ZERO]], %[[SIMD_ZERO]], %arg1, %arg1
  kgen.return %0, %1, %2, %3,
              %4, %5, %6, %7 : !pop.scalar<si32>, !pop.scalar<si32>,
                               !pop.scalar<si32>, !pop.scalar<si32>,
                               !pop.simd<2, si32>, !pop.simd<2, si32>,
                               !pop.simd<2, si32>, !pop.simd<2, si32>
}

// CHECK-LABEL: @div
kgen.func @div(%arg0: !pop.scalar<si64>, %arg1: !pop.simd<2, si32>, %arg2: !pop.simd<4, index>) -> (
    !pop.scalar<si4>, !pop.scalar<ui4>, !pop.scalar<f32>,
    !pop.scalar<si64>, !pop.simd<2, si32>, !pop.scalar<ui32>,
    !pop.simd<2, ui32>, !pop.scalar<index>, !pop.simd<4, index>
  ) {
  // CHECK-DAG: <si4> = <-3
  // CHECK-DAG: <ui4> = <0>
  // CHECK-DAG: <"1.25">
  // CHECK-DAG: %[[ONE_INDEX:.*]] = kgen.param.constant: scalar<index> = <1>
  // CHECK-DAG: %[[ONE_U32:.*]] = kgen.param.constant: scalar<ui32> = <1>
  // CHECK-DAG: %[[ONE_TWO:.*]] = kgen.param.constant: simd<2, ui32> = <<1, 2>>
  // CHECK-DAG: %[[ONE:.*]] = kgen.param.constant: scalar<si64> = <1>
  // CHECK-DAG: pop.shr %arg0, %[[ONE]] : !pop.scalar<si64>
  // CHECK-DAG: %[[TWO:.*]] = kgen.param.constant: simd<2, si32> = <2>
  // CHECK-DAG: pop.shr %arg1, %[[TWO]] : !pop.simd<2, si32>
  // CHECK-DAG: %[[U32:.*]] = pop.cast %{{.*}} : !pop.scalar<si64> to !pop.scalar<ui32>
  // CHECK-DAG: pop.shr %[[U32]], %[[ONE_U32]] : !pop.scalar<ui32>
  // CHECK-DAG: %[[SIMD_U32:.*]] = pop.cast %{{.*}} : !pop.simd<2, si32> to !pop.simd<2, ui32>
  // CHECK-DAG: pop.shr %[[SIMD_U32]], %[[ONE_TWO]] : !pop.simd<2, ui32>
  // CHECK-DAG: %[[INDEX:.*]] = pop.cast %{{.*}} : !pop.scalar<si64> to !pop.scalar<index>
  // CHECK-DAG: pop.shr %[[INDEX]], %[[ONE_INDEX]] : !pop.scalar<index>
  // CHECK-DAG: %[[SIMD_INDEX:.*]] = kgen.param.constant: simd<4, index> = <<1, 2, 3, 4>>
  // CHECK-DAG: pop.shr %arg2, %[[SIMD_INDEX]] : !pop.simd<4, index>
  %0 = kgen.param.constant: scalar<si4> = <7>
  %1 = kgen.param.constant: scalar<si4> = <-2>
  %2 = kgen.param.constant: scalar<ui4> = <7>
  %3 = kgen.param.constant: scalar<ui4> = <-2>
  %4 = kgen.param.constant: scalar<f32> = <"2.5">
  %5 = kgen.param.constant: scalar<f32> = <"2">
  %6 = pop.div %0, %1 : !pop.scalar<si4>
  %7 = pop.div %2, %3 : !pop.scalar<ui4>
  %8 = pop.div %4, %5 : !pop.scalar<f32>
  // Check the canonicalization of x / 2^n into x >> n
  %two_int = kgen.param.constant: scalar<si64> = <2>
  %9 = pop.div %arg0, %two_int : !pop.scalar<si64>
  %four_simd = kgen.param.constant: simd<2, si32> = <<4, 4>>
  %10 = pop.div %arg1, %four_simd : !pop.simd<2, si32>
  %two_uint = kgen.param.constant: scalar<ui32> = <2>
  %uarg0 = pop.cast %arg0 : !pop.scalar<si64> to !pop.scalar<ui32>
  %11 = pop.div %uarg0, %two_uint : !pop.scalar<ui32>
  %uarg1 = pop.cast %arg1 : !pop.simd<2, si32> to !pop.simd<2, ui32>
  %two_four_simd = kgen.param.constant: simd<2, ui32> = <<2, 4>>
  %12 = pop.div %uarg1, %two_four_simd : !pop.simd<2, ui32>
  %two_index = kgen.param.constant: scalar<index> = <2>
  %iarg0 = pop.cast %arg0 : !pop.scalar<si64> to !pop.scalar<index>
  %13 = pop.div %iarg0, %two_index : !pop.scalar<index>
  %14 = kgen.param.constant: simd<4, index> = <<2, 4, 8, 16>>
  %15 = pop.div %arg2, %14 : !pop.simd<4, index>
  kgen.return %6, %7, %8, %9, %10, %11, %12, %13, %15 : !pop.scalar<si4>, !pop.scalar<ui4>,
                                                   !pop.scalar<f32>, !pop.scalar<si64>,
                                                   !pop.simd<2, si32>, !pop.scalar<ui32>,
                                                   !pop.simd<2, ui32>, !pop.scalar<index>,
                                                   !pop.simd<4, index>
}

// CHECK-LABEL: @div_zero
kgen.func @div_zero() -> (!pop.scalar<si4>, !pop.scalar<f32>) {
  %0 = kgen.param.constant: scalar<si4> = <0>
  %1 = kgen.param.constant: scalar<f32> = <"0">
  // CHECK: pop.div
  %2 = pop.div %0, %0 : !pop.scalar<si4>
  // CHECK: pop.div
  %3 = pop.div %1, %1 : !pop.scalar<f32>
  kgen.return %2, %3 : !pop.scalar<si4>, !pop.scalar<f32>
}

// CHECK-LABEL: @rem
kgen.func @rem() -> (!pop.scalar<si4>, !pop.scalar<ui4>, !pop.scalar<f32>, !pop.scalar<f64>) {
  // CHECK-DAG: <si4> = <1
  // CHECK-DAG: <ui4> = <7>
  // CHECK-DAG: <"0.5">
  // CHECK-DAG: <"1.140{{.*}}">
  %0 = kgen.param.constant: scalar<si4> = <7>
  %1 = kgen.param.constant: scalar<si4> = <-2>
  %2 = kgen.param.constant: scalar<ui4> = <7>
  %3 = kgen.param.constant: scalar<ui4> = <-2>
  %4 = kgen.param.constant: scalar<f32> = <"2.5">
  %5 = kgen.param.constant: scalar<f32> = <"2">
  %6 = pop.rem %0, %1 : !pop.scalar<si4>
  %7 = pop.rem %2, %3 : !pop.scalar<ui4>
  %8 = pop.rem %4, %5 : !pop.scalar<f32>
  %9 = kgen.param.constant: scalar<f64> = <"3.14">
  %10 = kgen.param.constant: scalar<f64> = <"2.0">
  %11 = pop.rem %9, %10 : !pop.scalar<f64>
  kgen.return %6, %7, %8, %11 : !pop.scalar<si4>, !pop.scalar<ui4>, !pop.scalar<f32>, !pop.scalar<f64>
}

// CHECK-LABEL: @max
kgen.func @max() -> (!pop.scalar<si4>, !pop.scalar<f32>, !pop.scalar<f32>) {
  // CHECK-DAG: <-1>
  // CHECK-DAG: <"2">
  // CHECK-DAG: <"1.25">
  %0 = kgen.param.constant: scalar<si4> = <-2>
  %1 = kgen.param.constant: scalar<si4> = <-1>
  %2 = kgen.param.constant: scalar<f32> = <"1.25">
  %3 = kgen.param.constant: scalar<f32> = <"2">
  %4 = kgen.param.constant: scalar<f32> = <"NaN">
  %5 = pop.max %0, %1 : !pop.scalar<si4>
  %6 = pop.max %2, %3 : !pop.scalar<f32>
  %7 = pop.max %2, %4 : !pop.scalar<f32>
  kgen.return %5, %6, %7 : !pop.scalar<si4>, !pop.scalar<f32>, !pop.scalar<f32>
}

// CHECK-LABEL: @min
kgen.func @min() -> (!pop.scalar<ui4>, !pop.scalar<f32>, !pop.scalar<f32>) {
  // CHECK-DAG: <0>
  // CHECK-DAG: <"-2">
  // CHECK-DAG: <"1.25">
  %0 = kgen.param.constant: scalar<ui4> = <0>
  %1 = kgen.param.constant: scalar<ui4> = <-1>
  %2 = kgen.param.constant: scalar<f32> = <"1.25">
  %3 = kgen.param.constant: scalar<f32> = <"-2">
  %4 = kgen.param.constant: scalar<f32> = <"NaN">
  %5 = pop.min %0, %1 : !pop.scalar<ui4>
  %6 = pop.min %2, %3 : !pop.scalar<f32>
  %7 = pop.min %2, %4 : !pop.scalar<f32>
  kgen.return %5, %6, %7 : !pop.scalar<ui4>, !pop.scalar<f32>, !pop.scalar<f32>
}

// CHECK-LABEL: @min_max_identity
kgen.func @min_max_identity(%arg0 : !pop.scalar<ui4>, %arg1 : !pop.scalar<ui4>) ->
 (!pop.scalar<ui4>, !pop.scalar<ui4>, !pop.scalar<ui4>, !pop.scalar<ui4>) {
  // Confirm that folding does not erroneously act on min(x,y) and max(x,y)
  %0 = pop.min %arg0, %arg1 : !pop.scalar<ui4>
  %1 = pop.max %arg0, %arg1 : !pop.scalar<ui4>
  // min(x,x) -> x
  %2 = pop.min %arg0, %arg0 : !pop.scalar<ui4>
  // max(x,x) -> x
  %3 = pop.max %arg0, %arg0 : !pop.scalar<ui4>
  // CHECK: return %0, %1, %arg0, %arg0
  kgen.return %0, %1, %2, %3 : !pop.scalar<ui4>, !pop.scalar<ui4>, !pop.scalar<ui4>, !pop.scalar<ui4>
}

// CHECK-LABEL: @shl
kgen.func @shl() -> !pop.scalar<ui4> {
  // CHECK-NEXT: <12>
  %0 = kgen.param.constant: scalar<ui4> = <6>
  %1 = kgen.param.constant: scalar<ui4> = <1>
  %2 = pop.shl %0, %1 : !pop.scalar<ui4>
  kgen.return %2 : !pop.scalar<ui4>
}

// CHECK-LABEL: @shr
kgen.func @shr() -> (!pop.scalar<ui4>, !pop.scalar<si4>) {
  // CHECK-DAG: <3>
  // CHECK-DAG: <-4>
  %0 = kgen.param.constant: scalar<ui4> = <7>
  %1 = kgen.param.constant: scalar<ui4> = <1>
  %2 = kgen.param.constant: scalar<si4> = <-7>
  %3 = kgen.param.constant: scalar<si4> = <1>
  %4 = pop.shr %0, %1 : !pop.scalar<ui4>
  %5 = pop.shr %2, %3 : !pop.scalar<si4>
  kgen.return %4, %5 : !pop.scalar<ui4>, !pop.scalar<si4>
}

// CHECK-LABEL: @fma
kgen.func @fma() -> (!pop.scalar<si8>, !pop.scalar<f32>) {
  // CHECK-DAG: <6>
  // CHECK-DAG: <"8.75">
  %0 = kgen.param.constant: scalar<si8> = <2>
  %1 = kgen.param.constant: scalar<f32> = <"2.5">
  %2 = pop.fma %0, %0, %0 : !pop.scalar<si8>
  %3 = pop.fma %1, %1, %1 : !pop.scalar<f32>
  kgen.return %2, %3 : !pop.scalar<si8>, !pop.scalar<f32>
}


// CHECK-LABEL: @index_folds
kgen.func @index_folds() -> (!pop.scalar<index>, !pop.scalar<index>) {
  // COM: Index folds go through the same path as integer folds. We just need to
  // check that ops can fold for index dtypes and do not fold when the results
  // differ between 64-bit and 32-bit arithmetic.
  // CHECK-DAG: %[[DNF_LHS:.*]] = kgen{{.*}}<4294967298>
  // CHECK-DAG: %[[DNF_RHS:.*]] = kgen{{.*}}<1>
  // CHECK-DAG: %[[FOLDED:.*]] = kgen{{.*}}<4294967297>
  // CHECK: %[[R2:.*]] = pop.shr %[[DNF_LHS]], %[[DNF_RHS]]
  // CHECK-NEXT: return %[[FOLDED]], %[[R2]]
  %0 = kgen.param.constant: scalar<index> = <8589934594>
  %1 = kgen.param.constant: scalar<index> = <4294967298>
  %2 = kgen.param.constant: scalar<index> = <2>
  %3 = pop.div %0, %2 : !pop.scalar<index>
  %4 = pop.div %1, %2 : !pop.scalar<index>
  kgen.return %3, %4 : !pop.scalar<index>, !pop.scalar<index>
}

// CHECK-LABEL: @cmp_eq
kgen.func @cmp_eq() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <false, true>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp eq(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp eq(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_ne
kgen.func @cmp_ne() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <true, false>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp ne(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp ne(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_lt
kgen.func @cmp_lt() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <false>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp lt(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp lt(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_gt
kgen.func @cmp_gt() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <true, false>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp gt(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp gt(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_le
kgen.func @cmp_le() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <false, true>
  // CHECK-DAG: <true>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp le(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp le(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_ge
kgen.func @cmp_ge() -> (!pop.simd<2, bool>, !pop.scalar<bool>) {
  // CHECK-DAG: <true>
  // CHECK-DAG: <false>
  %0 = kgen.param.constant: simd<2, si8> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, si8> = <<-2, 2>>
  %2 = pop.cmp ge(%0, %1) : !pop.simd<2, si8>
  %3 = kgen.param.constant: scalar<f32> = <"1">
  %4 = kgen.param.constant: scalar<f32> = <"2">
  %5 = pop.cmp ge(%3, %4) : !pop.scalar<f32>
  kgen.return %2, %5 : !pop.simd<2, bool>, !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_index
kgen.func @cmp_index() -> !pop.scalar<bool> {
  // CHECK: pop.cmp
  %0 = kgen.param.constant: scalar<index> = <4294967296>
  %1 = kgen.param.constant: scalar<index> = <8589934592>
  %2 = pop.cmp eq(%0, %1) : !pop.scalar<index>
  kgen.return %2 : !pop.scalar<bool>
}

// CHECK-LABEL: @cmp_eq_self
kgen.func @cmp_eq_self(%simd: !pop.simd<2, si8>) -> !pop.simd<2, bool> {
  // CHECK-DAG: <true>
  %0 = pop.cmp eq(%simd, %simd) : !pop.simd<2, si8>
  kgen.return %0 : !pop.simd<2, bool>
}

// Floats cannot be removed symbolically as they could be NAN.
// CHECK-LABEL: @cmp_eq_self_float
kgen.func @cmp_eq_self_float(%simd: !pop.simd<2, f32>) -> !pop.simd<2, bool> {
  // CHECK-NEXT: %[[RES:.*]] = pop.cmp eq
  // CHECK-NEXT: kgen.return %[[RES]]
  %0 = pop.cmp eq(%simd, %simd) : !pop.simd<2, f32>
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @cmp_ne_self
kgen.func @cmp_ne_self(%simd: !pop.simd<2, si8>) -> !pop.simd<2, bool> {
  // CHECK-DAG: <false>
  %0 = pop.cmp ne(%simd, %simd) : !pop.simd<2, si8>
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @cmp_lt_self
kgen.func @cmp_lt_self(%simd: !pop.simd<2, si8>) -> !pop.simd<2, bool> {
  // CHECK-DAG: <false>
  %0 = pop.cmp lt(%simd, %simd) : !pop.simd<2, si8>
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @cmp_gt_self
kgen.func @cmp_gt_self(%simd: !pop.simd<2, si8>) -> !pop.simd<2, bool> {
  // CHECK-DAG: <false>
  %0 = pop.cmp gt(%simd, %simd) : !pop.simd<2, si8>
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @cmp_ge_self
kgen.func @cmp_ge_self(%simd: !pop.simd<2, si8>) -> !pop.simd<2, bool> {
  // CHECK-DAG: <true>
  %0 = pop.cmp ge(%simd, %simd) : !pop.simd<2, si8>
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @cmp_le_self
kgen.func @cmp_le_self(%simd: !pop.simd<2, si8>) -> !pop.simd<2, bool> {
  // CHECK-DAG: <true>
  %0 = pop.cmp le(%simd, %simd) : !pop.simd<2, si8>
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @cmp_true_false
kgen.func @cmp_true_false(%simd: !pop.simd<2, bool>) -> (!pop.simd<2, bool>, !pop.simd<2, bool>) {
  %true = kgen.param.constant: simd<2, bool> = <<true, true>>
  %false = kgen.param.constant: simd<2, bool> = <<false, false>>
  %0 = pop.cmp eq(%true, %simd) : <2, bool>
  %1 = pop.cmp ne(%simd, %false) : <2, bool>
  // CHECK-NEXT: return %arg0, %arg0
  kgen.return %0, %1 : !pop.simd<2, bool>, !pop.simd<2, bool>
}

// CHECK-LABEL: @cmp_unsigned
kgen.func @cmp_unsigned(%simd: !pop.simd<2, ui8>) -> (
    !pop.simd<2, bool>, !pop.simd<2, bool>,
    !pop.simd<2, bool>, !pop.simd<2, bool>,
    !pop.simd<2, bool>, !pop.simd<2, bool>,
    !pop.simd<2, bool>, !pop.simd<2, bool>
) {
  %zero = kgen.param.constant: simd<2, ui8> = <<0, 0>>

  // CHECK: %[[ZERO:.*]] = kgen.param.constant: simd<2, ui8> = <0>
  // CHECK: %[[TRUE:.*]] = kgen.param.constant: simd<2, bool> = <true>
  // CHECK: %[[FALSE:.*]] = kgen.param.constant: simd<2, bool> = <false>
  %0 = pop.cmp ge(%simd, %zero) : <2, ui8>
  %2 = pop.cmp gt(%zero, %simd) : <2, ui8>
  %5 = pop.cmp le(%zero, %simd) : <2, ui8>
  %7 = pop.cmp lt(%simd, %zero) : <2, ui8>

  // CHECK: %[[UNOPTIMIZED_GE:.*]] = pop.cmp ge(%[[ZERO]], %arg0)
  %1 = pop.cmp ge(%zero, %simd) : <2, ui8>
  // CHECK: %[[UNOPTIMIZED_GT:.*]] = pop.cmp gt(%arg0, %[[ZERO]])
  %3 = pop.cmp gt(%simd, %zero) : <2, ui8>
  // CHECK: %[[UNOPTIMIZED_LE:.*]] = pop.cmp le(%arg0, %[[ZERO]])
  %4 = pop.cmp le(%simd, %zero) : <2, ui8>
  // CHECK: %[[UNOPTIMIZED_LT:.*]] = pop.cmp lt(%[[ZERO]], %arg0)
  %6 = pop.cmp lt(%zero, %simd) : <2, ui8>

  // CHECK-NEXT: return %[[TRUE]], %[[UNOPTIMIZED_GE]], %[[FALSE]], %[[UNOPTIMIZED_GT]], %[[UNOPTIMIZED_LE]], %[[TRUE]], %[[UNOPTIMIZED_LT]], %[[FALSE]]
  kgen.return %0, %1, %2, %3,
              %4, %5, %6, %7 : !pop.simd<2, bool>, !pop.simd<2, bool>,
                               !pop.simd<2, bool>, !pop.simd<2, bool>,
                               !pop.simd<2, bool>, !pop.simd<2, bool>,
                               !pop.simd<2, bool>, !pop.simd<2, bool>
}

// CHECK-LABEL: @and
kgen.func @and() -> !pop.scalar<ui4> {
  // CHECK-NEXT <1>
  %0 = kgen.param.constant: scalar<ui4> = <7>
  %1 = kgen.param.constant: scalar<ui4> = <9>
  %2 = pop.simd.and %0, %1 : !pop.scalar<ui4>
  kgen.return %2 : !pop.scalar<ui4>
}

// CHECK-LABEL: @or
kgen.func @or() -> !pop.scalar<ui4> {
  // CHECK-NEXT <15>
  %0 = kgen.param.constant: scalar<ui4> = <6>
  %1 = kgen.param.constant: scalar<ui4> = <9>
  %2 = pop.simd.or %0, %1 : !pop.scalar<ui4>
  kgen.return %2 : !pop.scalar<ui4>
}

// CHECK-LABEL: @xor
kgen.func @xor() -> !pop.scalar<ui4> {
  // CHECK-NEXT <2>
  %0 = kgen.param.constant: scalar<ui4> = <5>
  %1 = kgen.param.constant: scalar<ui4> = <7>
  %2 = pop.simd.xor %0, %1 : !pop.scalar<ui4>
  kgen.return %2 : !pop.scalar<ui4>
}

// CHECK-LABEL: @xor_zero
kgen.func @xor_zero(%arg0: !pop.simd<2, ui16>) -> !pop.simd<2, ui16> {
  %0 = kgen.param.constant: simd<2, ui16> = <0>
  %1 = pop.simd.xor %0, %arg0 : !pop.simd<2, ui16>
  // CHECK-NEXT: return %arg0
  kgen.return %1 : !pop.simd<2, ui16>
}

// CHECK-LABEL: @not_not
kgen.func @not_not(%arg0: !pop.scalar<bool>) ->!pop.scalar<bool>{
  %0 = kgen.param.constant: scalar<bool> = <true>
  %1 = pop.simd.xor %arg0, %0 : !pop.scalar<bool>
  %2 = pop.simd.xor %1, %0 : !pop.scalar<bool>
  // CHECK-NEXT: return %arg0
  kgen.return %2 : !pop.scalar<bool>
}

// CHECK-LABEL: @bool_and
kgen.func @bool_and() -> i1 {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant: i1 = <1>
  %1 = kgen.param.constant: i1 = <1>
  // CHECK-NOT: pop.add
  %2 = pop.and %0, %1
  kgen.return %2 : i1
}

// CHECK-LABEL: @bool_and_true
kgen.func @bool_and_true(%arg0: i1) -> i1 {
  %0 = kgen.param.constant: i1 = <1>
  %1 = pop.and %arg0, %0
  // CHECK-NEXT: kgen.return %arg0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_and_false
kgen.func @bool_and_false(%arg0: i1) -> i1 {
  // CHECK-NEXT: <0>
  %0 = kgen.param.constant: i1 = <0>
  // CHECK-NOT: pop.add
  %1 = pop.and %arg0, %0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_or
kgen.func @bool_or() -> i1 {
  // CHECK-NEXT: <0>
  %0 = kgen.param.constant: i1 = <0>
  %1 = kgen.param.constant: i1 = <0>
  // CHECK-NOT: pop.or
  %2 = pop.or %0, %1
  kgen.return %2 : i1
}

// CHECK-LABEL: @bool_or_true
kgen.func @bool_or_true(%arg0: i1) -> i1 {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant: i1 = <1>
  // CHECK-NOT: pop.or
  %1 = pop.or %arg0, %0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_or_false
kgen.func @bool_or_false(%arg0: i1) -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  %1 = pop.or %arg0, %0
  // CHECK-NEXT: kgen.return %arg0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_xor
kgen.func @bool_xor() -> i1 {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant: i1 = <1>
  %1 = kgen.param.constant: i1 = <0>
  // CHECK-NOT: pop.xor
  %2 = pop.xor %0, %1
  kgen.return %2 : i1
}

// CHECK-LABEL: @bool_xor_false
kgen.func @bool_xor_false(%arg0: i1) -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  %1 = pop.xor %0, %arg0
  // CHECK-NEXT: return %arg0
  kgen.return %1 : i1
}

// CHECK-LABEL: @bool_not_not
kgen.func @bool_not_not(%arg0: i1) ->i1{
  %0 = kgen.param.constant: i1 = <1>
  %1 = pop.xor %arg0, %0
  %2 = pop.xor %1, %0
  // CHECK-NEXT: return %arg0
  kgen.return %2 : i1
}

// CHECK-LABEL: @mask_ones
kgen.func @mask_ones(%arg0: !pop.simd<2, ui4>) -> !pop.simd<2, ui4> {
  %0 = kgen.param.constant: simd<2, ui4> = <15>
  %1 = kgen.param.constant: simd<2, ui4> = <15>
  %2 = pop.simd.xor %arg0, %0 : !pop.simd<2, ui4>
  %3 = pop.simd.xor %2, %1 : !pop.simd<2, ui4>
  // CHECK-NEXT: return %arg0
  kgen.return %3 : !pop.simd<2, ui4>
}

// CHECK-LABEL: @simd_select
kgen.func @simd_select() -> !pop.simd<2, si4> {
  // CHECK-NEXT: <1, 4>
  %0 = kgen.param.constant: simd<2, si4> = <<1, 3>>
  %1 = kgen.param.constant: simd<2, si4> = <<2, 4>>
  %2 = kgen.param.constant: simd<2, bool> = <<true, false>>
  %3 = pop.simd.select %2, %0, %1 : !pop.simd<2, si4>
  kgen.return %3 : !pop.simd<2, si4>
}

// CHECK-LABEL: @simd_select_true_false
kgen.func @simd_select_true_false(%arg0: !pop.simd<2, bool>) -> !pop.simd<2, bool> {
  // CHECK-NEXT: return %arg0
  %true = kgen.param.constant: simd<2, bool> = <true>
  %false = kgen.param.constant: simd<2, bool> = <false>
  %0 = pop.simd.select %arg0, %true, %false : <2, bool>
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @simd_select_false_true
kgen.func @simd_select_false_true(%arg0: !pop.simd<2, bool>) -> !pop.simd<2, bool> {
  // CHECK-NEXT: %[[TRUE:.*]] = kgen.param.constant: simd<2, bool> = <true>
  // CHECK-NEXT: %0 = pop.simd.xor %arg0, %[[TRUE]]
  // CHECK-NEXT: return %0
  %true = kgen.param.constant: simd<2, bool> = <<true, true>>
  %false = kgen.param.constant: simd<2, bool> = <<false, false>>
  %0 = pop.simd.select %arg0, %false, %true : <2, bool>
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @simd_select_equal
kgen.func @simd_select_equal(%arg0: !pop.simd<2, bool>, %arg1: !pop.simd<2, bool>) -> !pop.simd<2, bool> {
  %0 = pop.simd.select %arg0, %arg1, %arg1 : <2, bool>
  // CHECK-NEXT: return %arg1
  kgen.return %0 : !pop.simd<2, bool>
}

// CHECK-LABEL: @simd_select_all_true
kgen.func @simd_select_all_true(%arg0: !pop.simd<2, f32>, %arg1: !pop.simd<2, f32>) -> !pop.simd<2, f32> {
  // CHECK: (%[[ARG0:.*]]: !pop.simd<2, f32>, %[[ARG1:.*]]: !pop.simd<2, f32>)
  // CHECK-NEXT: kgen.return %[[ARG0]]

  %true = kgen.param.constant: simd<2, bool> = <<true, true>>
  %0 = pop.simd.select %true, %arg0, %arg1 : <2, f32>
  kgen.return %0 : !pop.simd<2, f32>
}

// CHECK-LABEL: @simd_select_all_false
kgen.func @simd_select_all_false(%arg0: !pop.simd<2, f32>, %arg1: !pop.simd<2, f32>) -> !pop.simd<2, f32> {
  // CHECK: (%[[ARG0:.*]]: !pop.simd<2, f32>, %[[ARG1:.*]]: !pop.simd<2, f32>)
  // CHECK-NEXT: kgen.return %[[ARG1]]

  %true = kgen.param.constant: simd<2, bool> = <<false, false>>
  %0 = pop.simd.select %true, %arg0, %arg1 : <2, f32>
  kgen.return %0 : !pop.simd<2, f32>
}

// CHECK-LABEL: @bitcast
kgen.func @bitcast() -> (!pop.simd<2, bf16>, !pop.simd<2, f16>) {
  // CHECK-DAG: <"0.125", "8">
  // CHECK-DAG: <"5.9605E-8", "1.1921E-7">
  %0 = kgen.param.constant: simd<2, si16> = <<1, 2>>
  %1 = kgen.param.constant: simd<2, f16> = <<"1.5", "2.5">>
  %2 = pop.bitcast %0 : !pop.simd<2, si16> to !pop.simd<2, f16>
  %3 = pop.bitcast %1 : !pop.simd<2, f16> to !pop.simd<2, bf16>
  kgen.return %3, %2 : !pop.simd<2, bf16>, !pop.simd<2, f16>
}

// CHECK-LABEL: @bitcast_size_change
kgen.func @bitcast_size_change() -> (!pop.simd<4, si16>) {
  // CHECK: pop.bitcast
  %0 = kgen.param.constant: simd<2, si32> = <<1, 2>>
  %1 = pop.bitcast %0 : !pop.simd<2, si32> to !pop.simd<4, si16>
  kgen.return %1 : !pop.simd<4, si16>
}

// CHECK-LABEL: @bitcast_index
kgen.func @bitcast_index() -> (!pop.simd<2, ui64>, !pop.simd<2, index>,
                               !pop.simd<2, index>, !pop.simd<2, f64>) {
  %ui64_const = kgen.param.constant: simd<2, ui64> = <<1, 2>>
  %f64_const = kgen.param.constant: simd<2, f64> = <<"1.5", "2.5">>
  %index_const = kgen.param.constant: simd<2, index> = <<3, 4>>

  %2 = pop.bitcast %index_const : !pop.simd<2, index> to !pop.simd<2, ui64>
  %3 = pop.bitcast %ui64_const : !pop.simd<2, ui64> to !pop.simd<2, index>
  %4 = pop.bitcast %f64_const : !pop.simd<2, f64> to !pop.simd<2, index>
  %5 = pop.bitcast %index_const : !pop.simd<2, index> to !pop.simd<2, f64>
  // CHECK: [[SIMD_UI64:%.*]] = kgen.param.constant: simd<2, ui64> = <<3, 4>>
  // CHECK-NEXT: [[SIMD_INDEX:%.*]] = kgen.param.constant: simd<2, index> = <<1, 2>>
  // CHECK-NEXT: [[SIMD_INDEX_F64:%.*]] = kgen.param.constant: simd<2, index> = <<4609434218613702656, 4612811918334230528>>
  // CHECK-NEXT: [[SIMD_F64:%.*]] = kgen.param.constant: simd<2, f64> = <<"1.4821969375237396E-323", "1.9762625833649862E-323">>
  // CHECK: kgen.return [[SIMD_UI64]], [[SIMD_INDEX]], [[SIMD_INDEX_F64]], [[SIMD_F64]]
  kgen.return %2, %3, %4, %5 : !pop.simd<2, ui64>, !pop.simd<2, index>,
                               !pop.simd<2, index>, !pop.simd<2, f64>
}

// CHECK-LABEL: @pointer_bitcast
kgen.func @pointer_bitcast() -> !kgen.pointer<si32> {
  // CHECK-NEXT: pointer<si32> = <0>
  %0 = kgen.param.constant: pointer<si64> = <0>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<si64> to !kgen.pointer<si32>
  kgen.return %1 : !kgen.pointer<si32>
}

// CHECK-LABEL: @pointer_bitcast_of_bitcast
kgen.func @pointer_bitcast_of_bitcast(%arg0: !kgen.pointer<si32>) -> !kgen.pointer<f32> {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<si32> to !kgen.pointer<f32>
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<si32> to !kgen.pointer<f64>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<f64> to !kgen.pointer<f32>
  // CHECK-NEXT: return %0
  kgen.return %1 : !kgen.pointer<f32>
}

// CHECK-LABEL: @cast
kgen.func @cast() -> (
    !pop.scalar<f16>, !pop.scalar<f16>, !pop.scalar<f16>,
    !pop.scalar<ui8>, !pop.scalar<ui8>, !pop.scalar<ui8>,
    !pop.scalar<bool>, !pop.scalar<bool>, !pop.scalar<bool>,
    !pop.scalar<index>, !pop.scalar<index>, !pop.scalar<index>) {
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}<"10.125">
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}<"500">
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}<"1">
  // CHECK-DAG: %[[C3:.*]] = kgen{{.*}}ui8{{.*}}<10>
  // CHECK-DAG: %[[C4:.*]] = kgen{{.*}}ui8{{.*}}<244>
  // CHECK-DAG: %[[C5:.*]] = kgen{{.*}}ui8{{.*}}<1>
  // CHECK-DAG: %[[TRUE:.*]] = kgen{{.*}}<true>
  // CHECK-DAG: %[[C6:.*]] = kgen{{.*}}index{{.*}}<10>
  // CHECK-DAG: %[[C7:.*]] = kgen{{.*}}index{{.*}}<500>
  // CHECK-DAG: %[[C8:.*]] = kgen{{.*}}index{{.*}}<1>
  %0 = kgen.param.constant: scalar<bf16> = <"10.125">
  %1 = kgen.param.constant: scalar<si32> = <500>
  %2 = kgen.param.constant: scalar<bool> = <true>

  %3 = pop.cast %0 : !pop.scalar<bf16> to !pop.scalar<f16>
  %4 = pop.cast %1 : !pop.scalar<si32> to !pop.scalar<f16>
  %5 = pop.cast %2 : !pop.scalar<bool> to !pop.scalar<f16>

  %6 = pop.cast %0 : !pop.scalar<bf16> to !pop.scalar<ui8>
  %7 = pop.cast %1 : !pop.scalar<si32> to !pop.scalar<ui8>
  %8 = pop.cast %2 : !pop.scalar<bool> to !pop.scalar<ui8>

  %9 = pop.cast %0 : !pop.scalar<bf16> to !pop.scalar<bool>
  %10 = pop.cast %1 : !pop.scalar<si32> to !pop.scalar<bool>
  %11 = pop.cast %2 : !pop.scalar<bool> to !pop.scalar<bool>

  %12 = pop.cast %0 : !pop.scalar<bf16> to !pop.scalar<index>
  %13 = pop.cast %1 : !pop.scalar<si32> to !pop.scalar<index>
  %14 = pop.cast %2 : !pop.scalar<bool> to !pop.scalar<index>

  // CHECK-NEXT: return %[[C0]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[TRUE]], %[[TRUE]], %[[TRUE]]
  // CHECK-SAME: %[[C6]], %[[C7]], %[[C8]]
  kgen.return %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14 :
    !pop.scalar<f16>, !pop.scalar<f16>, !pop.scalar<f16>,
    !pop.scalar<ui8>, !pop.scalar<ui8>, !pop.scalar<ui8>,
    !pop.scalar<bool>, !pop.scalar<bool>, !pop.scalar<bool>,
    !pop.scalar<index>, !pop.scalar<index>, !pop.scalar<index>
}

// CHECK-LABEL: @cast_si64_ui64_index
kgen.func @cast_si64_ui64_index(%in : !pop.scalar<si64>) -> !pop.scalar<index> {
  // CHECK-NEXT: pop.cast {{.*}} !pop.scalar<si64> to !pop.scalar<ui64>
  %1 = pop.cast %in : !pop.scalar<si64> to !pop.scalar<ui64>
  // CHECK-NEXT: pop.cast {{.*}} !pop.scalar<ui64> to !pop.scalar<index>
  %2 = pop.cast %1 : !pop.scalar<ui64> to !pop.scalar<index>

  kgen.return %2 : !pop.scalar<index>
}

// CHECK-LABEL: @cast_fp_too_big
kgen.func @cast_fp_too_big() -> (!pop.scalar<si2>, !pop.scalar<si2>) {
  // CHECK-DAG: <1>
  // CHECK-DAG: <"7">
  %0 = kgen.param.constant: scalar<f16> = <"1.5">
  %1 = kgen.param.constant: scalar<f16> = <"7">
  %2 = pop.cast %0 : !pop.scalar<f16> to !pop.scalar<si2>
  // CHECK: %[[TOO_BIG:.*]] = pop.cast
  %3 = pop.cast %1 : !pop.scalar<f16> to !pop.scalar<si2>
  // CHECK-NEXT: return %{{.*}}, %[[TOO_BIG]]
  kgen.return %2, %3 : !pop.scalar<si2>, !pop.scalar<si2>
}

// CHECK-LABEL: @cast_index_to_int
kgen.func @cast_index_to_int() -> !pop.scalar<si32> {
  // CHECK-NEXT: scalar<si32> = <-2>
  %0 = kgen.param.constant: scalar<index> = <-2>
  %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si32>
  kgen.return %1 : !pop.scalar<si32>
}

// CHECK-LABEL: @cast_index_to_index
kgen.func @cast_index_to_index() -> !pop.scalar<index> {
  // CHECK-NEXT: scalar<index> = <99999999999>
  // CHECK-NOT: pop.cast
  %0 = kgen.param.constant: scalar<index> = <99999999999>
  %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<index>
  kgen.return %1 : !pop.scalar<index>
}

// CHECK-LABEL: @cast_index_to_si64_large
kgen.func @cast_index_to_si64_large() -> !pop.scalar<si64> {
  // CHECK-NEXT: kgen.param.constant: scalar<index> = <-8664705627211539068>
  %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
  %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si64>
  kgen.return %1 : !pop.scalar<si64>
}

// CHECK-LABEL: @cast_index_to_si32_large
kgen.func @cast_index_to_si32_large() -> !pop.scalar<si32> {
  // CHECK-NEXT: kgen.param.constant: scalar<si32> = <-1095082620>
  %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
  %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si32>
  kgen.return %1 : !pop.scalar<si32>
}

// CHECK-LABEL: @cast_and_truncate
kgen.func @cast_and_truncate(%v0 : !pop.simd<2, si64>) -> !pop.simd<2, si32> {
  // CHECK-NEXT: pop.cast %arg0 : !pop.simd<2, si64> to !pop.simd<2, si32>
  // CHECK-NOT: pop.cast
  %v1 = pop.cast %v0 : !pop.simd<2, si64> to !pop.simd<2, si32>
  %v2 = pop.cast %v1 : !pop.simd<2, si32> to !pop.simd<2, si64>
  %v3 = pop.cast %v2 : !pop.simd<2, si64> to !pop.simd<2, si32>
  kgen.return %v3 : !pop.simd<2, si32>
}

// CHECK-LABEL: @sext_and_sext
kgen.func @sext_and_sext(%v0 : !pop.simd<2, si16>) -> !pop.simd<2, si64> {
  // CHECK-NEXT: pop.cast %arg0 : !pop.simd<2, si16> to !pop.simd<2, si64>
  %v1 = pop.cast %v0 : !pop.simd<2, si16> to !pop.simd<2, si32>
  %v2 = pop.cast %v1 : !pop.simd<2, si32> to !pop.simd<2, si64>
  kgen.return %v2 : !pop.simd<2, si64>
}

// CHECK-LABEL: @zext_and_zext
kgen.func @zext_and_zext(%v0 : !pop.simd<2, ui16>) -> !pop.simd<2, ui64> {
  // CHECK-NEXT: pop.cast %arg0 : !pop.simd<2, ui16> to !pop.simd<2, ui64>
  %v1 = pop.cast %v0 : !pop.simd<2, ui16> to !pop.simd<2, ui32>
  %v2 = pop.cast %v1 : !pop.simd<2, ui32> to !pop.simd<2, ui64>
  kgen.return %v2 : !pop.simd<2, ui64>
}

// CHECK-LABEL: @fpext_and_fptoint
kgen.func @fpext_and_fptoint(%arg0 : !pop.simd<2, f16>) -> (!pop.simd<2, si32>, !pop.simd<2, ui32>) {
  // COM: fptosi(fpext)
  %v12 = pop.cast %arg0 : !pop.simd<2, f16> to !pop.simd<2, f32>
  %v13 = pop.cast %v12 : !pop.simd<2, f32> to !pop.simd<2, si32>

  // COM: fptoui(fpext)
  %v14 = pop.cast %arg0 : !pop.simd<2, f16> to !pop.simd<2, f32>
  %v15 = pop.cast %v14 : !pop.simd<2, f32> to !pop.simd<2, ui32>

  kgen.return %v13, %v15 : !pop.simd<2, si32>, !pop.simd<2, ui32>
}

// CHECK-LABEL: @intext_and_fptoint
kgen.func @intext_and_fptoint(%arg0 : !pop.simd<2, ui16>, %arg1 : !pop.simd<2, si16>) -> (!pop.simd<2, f32>, !pop.simd<2, f32>) {
  // COM: uitofp(zext)
  %v12 = pop.cast %arg0 : !pop.simd<2, ui16> to !pop.simd<2, ui32>
  %v13 = pop.cast %v12 : !pop.simd<2, ui32> to !pop.simd<2, f32>

  // COM: sitofp(sext)
  %v14 = pop.cast %arg1 : !pop.simd<2, si16> to !pop.simd<2, si32>
  %v15 = pop.cast %v14 : !pop.simd<2, si32> to !pop.simd<2, f32>

  kgen.return %v13, %v15 : !pop.simd<2, f32>, !pop.simd<2, f32>
}

// CHECK-LABEL: @unsupported_intcast_and_intcast
kgen.func @unsupported_intcast_and_intcast(%v0 : !pop.simd<2, ui8>) -> !pop.simd<2, si64> {
  // CHECK-NEXT: pop.cast %arg0 : !pop.simd<2, ui8> to !pop.simd<2, ui16>
  %v1 = pop.cast %v0 : !pop.simd<2, ui8> to !pop.simd<2, ui16>
  // CHECK-NEXT: pop.cast %0 : !pop.simd<2, ui16> to !pop.simd<2, si32>
  %v2 = pop.cast %v1 : !pop.simd<2, ui16> to !pop.simd<2, si32>
  // CHECK-NEXT: pop.cast %1 : !pop.simd<2, si32> to !pop.simd<2, ui64>
  %v3 = pop.cast %v2 : !pop.simd<2, si32> to !pop.simd<2, ui64>
  // CHECK-NEXT: pop.cast %2 : !pop.simd<2, ui64> to !pop.simd<2, si64>
  %v4 = pop.cast %v3 : !pop.simd<2, ui64> to !pop.simd<2, si64>
  kgen.return %v4 : !pop.simd<2, si64>
}

// CHECK-LABEL: @fpext_and_fpext
kgen.func @fpext_and_fpext(%arg0 : !pop.simd<2, f16>) -> !pop.simd<2, f64> {
  // COM: fpext(fpext)
  %v8 = pop.cast %arg0 : !pop.simd<2, f16> to !pop.simd<2, f32>
  // CHECK-NEXT: pop.cast %arg0 : !pop.simd<2, f16> to !pop.simd<2, f64>
  %v9 = pop.cast %v8 : !pop.simd<2, f32> to !pop.simd<2, f64>
  kgen.return %v9 : !pop.simd<2, f64>
}

// CHECK-LABEL: @fpext_and_fptrunc
kgen.func @fpext_and_fptrunc(%arg0 : !pop.simd<2, f16>) -> !pop.simd<2, f32> {
  // COM: fptrunc(fpext)
  %v10 = pop.cast %arg0 : !pop.simd<2, f16> to !pop.simd<2, f64>
  // CHECK-NEXT: pop.cast %arg0 : !pop.simd<2, f16> to !pop.simd<2, f32>
  %v11 = pop.cast %v10 : !pop.simd<2, f64> to !pop.simd<2, f32>
  kgen.return %v11 : !pop.simd<2, f32>
}

// COM: test chain of int2fp/fp2int conversions
// CHECK-LABEL: @unsupported_conv_conv
kgen.func @unsupported_conv_conv(%arg0 : !pop.simd<2, ui16>, %arg1 : !pop.simd<2, si16>) -> (!pop.simd<2, ui32>, !pop.simd<2, f64>, !pop.simd<2, si32>, !pop.simd<2, f64>, !pop.simd<2, f16>) {
  // COM: fptoui(uitofp)
  // CHECK-NEXT: pop.cast %arg0 : !pop.simd<2, ui16> to !pop.simd<2, f32>
  %0 = pop.cast %arg0 : !pop.simd<2, ui16> to !pop.simd<2, f32>
  // CHECK: pop.cast %0 : !pop.simd<2, f32> to !pop.simd<2, ui32>
  %v2 = pop.cast %0 : !pop.simd<2, f32> to !pop.simd<2, ui32>

  // COM: fpext(uitofp)
  // CHECK-NEXT: pop.cast %arg0 : !pop.simd<2, ui16> to !pop.simd<2, f32>
  %1 = pop.cast %arg0 : !pop.simd<2, ui16> to !pop.simd<2, f32>
  // CHECK-NEXT: pop.cast %2 : !pop.simd<2, f32> to !pop.simd<2, f64>
  %v3 = pop.cast %1 : !pop.simd<2, f32> to !pop.simd<2, f64>

  // COM: fptrunc(sitofp)
  // CHECK-NEXT: pop.cast %arg1 : !pop.simd<2, si16> to !pop.simd<2, f32>
  %2 = pop.cast %arg1 : !pop.simd<2, si16> to !pop.simd<2, f32>
  // CHECK-NEXT: pop.cast %4 : !pop.simd<2, f32> to !pop.simd<2, si32>
  %v5 = pop.cast %2 : !pop.simd<2, f32> to !pop.simd<2, si32>

  // COM: fpext(sitofp)
  // CHECK-NEXT: pop.cast %arg1 : !pop.simd<2, si16> to !pop.simd<2, f32>
  %3 = pop.cast %arg1 : !pop.simd<2, si16> to !pop.simd<2, f32>
  // CHECK-NEXT: pop.cast %6 : !pop.simd<2, f32> to !pop.simd<2, f64>
  %v6 = pop.cast %3 : !pop.simd<2, f32> to !pop.simd<2, f64>

  // COM: fptrunc(sitofp)
  // CHECK-NEXT: pop.cast %arg1 : !pop.simd<2, si16> to !pop.simd<2, f32>
  %4 = pop.cast %arg1 : !pop.simd<2, si16> to !pop.simd<2, f32>
  // CHECK: pop.cast %8 : !pop.simd<2, f32> to !pop.simd<2, f16>
  %v7 = pop.cast %4 : !pop.simd<2, f32> to !pop.simd<2, f16>

  kgen.return %v2, %v3, %v5, %v6, %v7 : !pop.simd<2, ui32>, !pop.simd<2, f64>, !pop.simd<2, si32>, !pop.simd<2, f64>, !pop.simd<2, f16>
}

// CHECK-LABEL: @simd_extractelement
kgen.func @simd_extractelement() -> (!pop.scalar<si8>) {
  // CHECK-NEXT: <20>
  %idx1 = index.constant 1
  %0 = kgen.param.constant: simd<2, si8> = <<10, 20>>
  %1 = pop.simd.extractelement %0[%idx1] : !pop.simd<2, si8>
  kgen.return %1 : !pop.scalar<si8>
}

// CHECK-LABEL: @simd_extractelement_scalar
kgen.func @simd_extractelement_scalar(%scalar : !pop.scalar<f32>, %arg : index) -> (!pop.scalar<f32>) {
  // CHECK: (%[[SCALAR:.*]]: !pop.scalar<f32>, %[[ARG:.*]]: index)
  // CHECK-NEXT: kgen.return %[[SCALAR]]
  %1 = pop.simd.extractelement %scalar[%arg] : !pop.scalar<f32>
  kgen.return %1 : !pop.scalar<f32>
}

// CHECK-LABEL: @simd_insertelement
kgen.func @simd_insertelement() -> (!pop.simd<2, si8>) {
  // CHECK-NEXT: <30, 20>
  %idx0 = index.constant 0
  %0 = kgen.param.constant: simd<2, si8> = <<10, 20>>
  %1 = kgen.param.constant: scalar<si8> = <30>
  %2 = pop.simd.insertelement %1, %0[%idx0] : !pop.simd<2, si8>
  kgen.return %2 : !pop.simd<2, si8>
}

// CHECK-LABEL: @simd_insertelement2
kgen.func @simd_insertelement2() -> (!pop.simd<2, si8>) {
  // CHECK-NEXT: <30, 0>
  %idx0 = index.constant 0

  // Start from unknown/undef instead of constant.
  %0 = kgen.param.constant: simd<2, si8> = <#kgen.unknown : !pop.simd<2, si8>>
  %1 = kgen.param.constant: scalar<si8> = <30>
  %2 = pop.simd.insertelement %1, %0[%idx0] : !pop.simd<2, si8>
  kgen.return %2 : !pop.simd<2, si8>
}

// CHECK-LABEL: @simd_shuffle
kgen.func @simd_shuffle() -> !pop.simd<2, si8> {
  // CHECK-NEXT: <3, 2>
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = kgen.param.constant: scalar<si8> = <<3>>
  %2 = pop.simd.shuffle <1, si8> %0, %1 -> <2, si8> :array<2, index> [1, 0]
  kgen.return %2 : !pop.simd<2, si8>
}

// CHECK-LABEL: @simd_splat_scalar
kgen.func @simd_splat_scalar(%arg0: !pop.scalar<si8>) -> !pop.scalar<si8> {
  // CHECK: (%[[ARG0:.*]]: !pop.scalar<si8>)
  // CHECK-NEXT: kgen.return %[[ARG0]]
  %1 = pop.simd.splat %arg0 : !pop.scalar<si8>
  kgen.return %1 : !pop.scalar<si8>
}

// CHECK-LABEL: @simd_splat
kgen.func @simd_splat() -> !pop.simd<2, si8> {
  // CHECK-NEXT: <2>
  %0 = kgen.param.constant: scalar<si8> = <<2>>
  %1 = pop.simd.splat %0 : !pop.simd<2, si8>
  kgen.return %1 : !pop.simd<2, si8>
}

// CHECK-LABEL: @array_create
kgen.func @array_create() -> !pop.array<2, index> {
  // CHECK-NEXT: constant: array<2, index> = <[0, 0]>
  %idx0 = index.constant 0
  %0 = pop.array.create [%idx0, %idx0] : !pop.array<2, index>
  kgen.return %0 : !pop.array<2, index>
}

// CHECK-LABEL: @array_repeat
kgen.func @array_repeat() -> !pop.array<3, index> {
  // CHECK-NEXT: constant: array<3, index> = <[0, 1, 0]>
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = pop.array.repeat [%idx0, %idx1] : !pop.array<3, index>
  kgen.return %0 : !pop.array<3, index>
}

// CHECK-LABEL: @array_repeat_get
kgen.func @array_repeat_get(%arg0: index, %arg1: index) -> (index, index, index) {
  // CHECK: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
  // CHECK-NEXT: kgen.return %[[ARG0]], %[[ARG1]], %[[ARG0]] : index, index, index

  %0 = pop.array.repeat [%arg0, %arg1] : !pop.array<3, index>
  %1 = pop.array.get %0[0] : !pop.array<3, index>
  %2 = pop.array.get %0[1] : !pop.array<3, index>
  %3 = pop.array.get %0[2] : !pop.array<3, index>
  kgen.return %1, %2, %3 : index, index, index
}

// CHECK-LABEL: @array_get
kgen.func @array_get() -> index {
  // CHECK-NEXT: constant = <1>
  %0 = kgen.param.constant: array<2, index> = <[0, 1]>
  %1 = pop.array.get %0[1] : !pop.array<2, index>
  kgen.return %1 : index
}

// CHECK-LABEL: @array_get_unknown
kgen.func @array_get_unknown() -> index {
  %array = kgen.param.constant: array<4, index> = <#interp.uninitmem>
  %0 = pop.array.get %array[0] : !pop.array<4, index>

  // CHECK: %[[OUT:.*]] = kgen.param.constant = <#interp.uninitmem>
  // CHECK-NEXT: kgen.return %[[OUT]]

  kgen.return %0 : index
}

// CHECK-LABEL: @array_get_non_const_0
kgen.func @array_get_non_const_0(%arg0: index, %arg1: index) -> index {
  // CHECK: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
  // CHECK: kgen.return %[[ARG0]]
  %0 = pop.array.create [%arg0, %arg1] : !pop.array<2, index>
  %1 = pop.array.get %0[0] : !pop.array<2, index>
  kgen.return %1 : index
}

// CHECK-LABEL: @array_get_non_const_1
kgen.func @array_get_non_const_1(%arg0: index, %arg1: index) -> index {
  // CHECK: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
  // CHECK: kgen.return %[[ARG1]]
  %0 = pop.array.create [%arg0, %arg1] : !pop.array<2, index>
  %1 = pop.array.get %0[1] : !pop.array<2, index>
  kgen.return %1 : index
}

// CHECK-LABEL: @array_gep
kgen.func @array_gep(%array: !kgen.pointer<array<1, index>>, %idx: index) -> !kgen.pointer<index> {
  // CHECK: (%[[ARRAY:.*]]: !kgen.pointer<array<1, index>>, %[[IDX:.*]]: index)
  // CHECK-NEXT: %[[ZERO:.*]] = kgen.param.constant = <0>
  // CHECK-NEXT: %[[GEP:.*]] = pop.array.gep %[[ARRAY]][%[[ZERO]]]
  // CHECK-NEXT: kgen.return %[[GEP]]
  %1 = pop.array.gep %array[%idx] : !kgen.pointer<array<1, index>>
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @array_gep_unchanged
kgen.func @array_gep_unchanged(%array: !kgen.pointer<array<2, index>>, %idx: index) -> !kgen.pointer<index> {
  // CHECK: (%[[ARRAY:.*]]: !kgen.pointer<array<2, index>>, %[[IDX:.*]]: index)
  // CHECK-NEXT: %[[GEP:.*]] = pop.array.gep %[[ARRAY]][%[[IDX]]]
  // CHECK-NEXT: kgen.return %[[GEP]]
  %1 = pop.array.gep %array[%idx] : !kgen.pointer<array<2, index>>
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @array_replace
kgen.func @array_replace() -> !pop.array<2, index> {
  // CHECK-NEXT: constant: array<2, index> = <[0, 1]>
  %0 = kgen.param.constant: array<2, index> = <[0, 0]>
  %1 = index.constant 1
  %2 = pop.array.replace %1, %0[1] : !pop.array<2, index>
  kgen.return %2 : !pop.array<2, index>
}

kgen.func @array_replace_with_create(%idx0: index, %idx1: index, %idx2: index) -> !pop.array<3, index> {
  // CHECK: (%[[IDX0:.*]]: index, %[[IDX1:.*]]: index, %[[IDX2:.*]]: index)
  // CHECK-NEXT: %[[OUT:.*]] = pop.array.create [%[[IDX0]], %[[IDX1]], %[[IDX2]]]
  // CHECK-NEXT: kgen.return %[[OUT]]
  %0 = pop.array.create [%idx0, %idx0, %idx0] : !pop.array<3, index>
  %1 = pop.array.replace %idx1, %0[1] : !pop.array<3, index>
  %2 = pop.array.replace %idx2, %1[2] : !pop.array<3, index>
  kgen.return %2 : !pop.array<3, index>
}

kgen.func @array_replace_with_const(%idx0: index, %idx1: index) -> !pop.array<3, index> {
  // CHECK: (%[[IDX0:.*]]: index, %[[IDX1:.*]]: index)
  // CHECK-NEXT: %[[C0:.*]] = kgen.param.constant = <0>
  // CHECK-NEXT: %[[OUT:.*]] = pop.array.create [%[[C0]], %[[IDX0]], %[[IDX1]]]
  // CHECK-NEXT: kgen.return %[[OUT]]
  %0 = kgen.param.constant: array<3, index> = <[0, 0, 0]>
  %1 = pop.array.replace %idx0, %0[1] : !pop.array<3, index>
  %2 = pop.array.replace %idx1, %1[2] : !pop.array<3, index>
  kgen.return %2 : !pop.array<3, index>
}

// CHECK-LABEL: @pointer_to_index
kgen.func @pointer_to_index() -> index {
  // CHECK-DAG: <1>
  %0 = kgen.param.constant: pointer<i8> = <#interp.pointer<1>>
  %1 = pop.pointer_to_index %0 : !kgen.pointer<i8>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @bitcast_ptr_to_index
kgen.func @bitcast_ptr_to_index(%arg0: !kgen.pointer<none>) -> index {
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: %0 = pop.pointer_to_index %arg0
  %1 = pop.pointer_to_index %0 : !kgen.pointer<index>
  // CHECK-NEXT: return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @cast_to_builtin
kgen.func @cast_to_builtin() -> (
    vector<2xi1>, vector<2xindex>, vector<2xi4>, vector<2xbf16>,
    i1, index, ui8, f16) {
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}vector<2xi1> = <#M.dense_array<true, false>>
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}vector<2xindex> = <#M.dense_array<1, 2>>
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}vector<2xi4> = <#M.dense_array<3, 4>>
  // CHECK-DAG: %[[C3:.*]] = kgen{{.*}}vector<2xbf16> = <#M.dense_array<1.5{{0+}}e+00, 2.5{{0+}}e+00>>
  // CHECK-DAG: %[[C4:.*]] = kgen{{.*}}i1 = <1>
  // CHECK-DAG: %[[C5:.*]] = kgen{{.*}}constant = <10>
  // CHECK-DAG: %[[C6:.*]] = kgen{{.*}}ui8 = <66>
  // CHECK-DAG: %[[C7:.*]] = kgen{{.*}}f16 = <5.5{{0+}}e+00>
  %0 = kgen.param.constant: simd<2, bool> = <<true, false>>
  %1 = kgen.param.constant: simd<2, index> = <<1, 2>>
  %2 = kgen.param.constant: simd<2, si4> = <<3, 4>>
  %3 = kgen.param.constant: simd<2, bf16> = <<"1.5", "2.5">>
  %4 = kgen.param.constant: scalar<bool> = <<true>>
  %5 = kgen.param.constant: scalar<index> = <<10>>
  %6 = kgen.param.constant: scalar<ui8> = <<66>>
  %7 = kgen.param.constant: scalar<f16> = <<"5.5">>

  %a0 = pop.cast_to_builtin %0 : !pop.simd<2, bool> to vector<2xi1>
  %a1 = pop.cast_to_builtin %1 : !pop.simd<2, index> to vector<2xindex>
  %a2 = pop.cast_to_builtin %2 : !pop.simd<2, si4> to vector<2xi4>
  %a3 = pop.cast_to_builtin %3 : !pop.simd<2, bf16> to vector<2xbf16>
  %a4 = pop.cast_to_builtin %4 : !pop.scalar<bool> to i1
  %a5 = pop.cast_to_builtin %5 : !pop.scalar<index> to index
  %a6 = pop.cast_to_builtin %6 : !pop.scalar<ui8> to ui8
  %a7 = pop.cast_to_builtin %7 : !pop.scalar<f16> to f16

  // CHECK: return %[[C0]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C7]]
  kgen.return %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7 :
    vector<2xi1>, vector<2xindex>, vector<2xi4>, vector<2xbf16>,
    i1, index, ui8, f16
}

// CHECK-LABEL: @cast_from_builtin
kgen.func @cast_from_builtin() -> (
    !pop.simd<2, bool>, !pop.simd<2, index>, !pop.simd<2, si4>, !pop.simd<2, bf16>,
    !pop.scalar<bool>, !pop.scalar<index>, !pop.scalar<ui8>, !pop.scalar<f16>,
    !pop.scalar<index>) {
  // CHECK-DAG: %[[C0:.*]] = kgen{{.*}}simd<2, bool> = <<true, false>>
  // CHECK-DAG: %[[C1:.*]] = kgen{{.*}}simd<2, index> = <<1, 2>>
  // CHECK-DAG: %[[C2:.*]] = kgen{{.*}}simd<2, si4> = <<3, 4>>
  // CHECK-DAG: %[[C3:.*]] = kgen{{.*}}simd<2, bf16> = <<"1.5", "2.5">>
  // CHECK-DAG: %[[C4:.*]] = kgen{{.*}}scalar<bool> = <true>
  // CHECK-DAG: %[[C5:.*]] = kgen{{.*}}scalar<index> = <10>
  // CHECK-DAG: %[[C6:.*]] = kgen{{.*}}scalar<ui8> = <66>
  // CHECK-DAG: %[[C7:.*]] = kgen{{.*}}scalar<f16> = <"5.5">
  // CHECK-DAG: %[[C8:.*]] = kgen{{.*}}scalar<index> = <8>
  %0 = kgen.param.constant: vector<2xi1> = <#M.dense_array<true, false>>
  %1 = kgen.param.constant: vector<2xindex> = <#M.dense_array<1, 2>>
  %2 = kgen.param.constant: vector<2xi4> = <#M.dense_array<3, 4>>
  %3 = kgen.param.constant: vector<2xbf16> = <#M.dense_array<1.5, 2.5>>
  %4 = kgen.param.constant: i1 = <1>
  %5 = kgen.param.constant = <10>
  %6 = kgen.param.constant: ui8 = <66>
  %7 = kgen.param.constant: f16 = <5.5>
  %8 = index.constant 8

  %a0 = pop.cast_from_builtin %0 : vector<2xi1> to !pop.simd<2, bool>
  %a1 = pop.cast_from_builtin %1 : vector<2xindex> to !pop.simd<2, index>
  %a2 = pop.cast_from_builtin %2 : vector<2xi4> to !pop.simd<2, si4>
  %a3 = pop.cast_from_builtin %3 : vector<2xbf16> to !pop.simd<2, bf16>
  %a4 = pop.cast_from_builtin %4 : i1 to !pop.scalar<bool>
  %a5 = pop.cast_from_builtin %5 : index to !pop.scalar<index>
  %a6 = pop.cast_from_builtin %6 : ui8 to !pop.scalar<ui8>
  %a7 = pop.cast_from_builtin %7 : f16 to !pop.scalar<f16>
  %a8 = pop.cast_from_builtin %8 : index to !pop.scalar<index>

  // CHECK: return %[[C0]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C7]], %[[C8]]
  kgen.return %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8 :
    !pop.simd<2, bool>, !pop.simd<2, index>, !pop.simd<2, si4>, !pop.simd<2, bf16>,
    !pop.scalar<bool>, !pop.scalar<index>, !pop.scalar<ui8>, !pop.scalar<f16>, !pop.scalar<index>
}

// CHECK-LABEL: @cast_from_parameter
kgen.generator @cast_from_parameter<N>() -> !pop.scalar<index> {
  %0 = kgen.param.constant = <N>
  // CHECK: pop.cast_from_builtin
  %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
  kgen.return %1 : !pop.scalar<index>
}

// CHECK-LABEL: @variadic_create(
kgen.func @variadic_create() -> !kgen.variadic<index> {
  // CHECK-NEXT: kgen.param.constant{{.*}} = <[13, 17]>
  %0 = index.constant 13
  %1 = index.constant 17
  %2 = pop.variadic.create [%0, %1] : !kgen.variadic<index>
  kgen.return %2 : !kgen.variadic<index>
}


// CHECK-LABEL: @variadic_create_to_splat(
kgen.func @variadic_create_to_splat(%a: index) -> !kgen.variadic<index> {
  // CHECK-NEXT: %0 = pop.variadic.splat 2, %arg0 : !kgen.variadic<index>
  %2 = pop.variadic.create [%a, %a] : !kgen.variadic<index>
  kgen.return %2 : !kgen.variadic<index>
}

// CHECK-LABEL: @variadic_splat_cst(
kgen.func @variadic_splat_cst() -> !kgen.variadic<index> {
  // CHECK-NEXT: %variadic = kgen.param.constant: variadic<index> = <[13, 13, 13, 13]>
  %0 = index.constant 13
  %1 = pop.variadic.splat 4, %0 : !kgen.variadic<index>
  kgen.return %1 : !kgen.variadic<index>
}

// CHECK-LABEL: @variadic_splat_zero(
kgen.func @variadic_splat_zero(%0: index) -> !kgen.variadic<index> {
  // CHECK-NEXT: %variadic = kgen.param.constant: variadic<index> = <[]>
  %1 = pop.variadic.splat 0, %0 : !kgen.variadic<index>
  kgen.return %1 : !kgen.variadic<index>
}

// CHECK-LABEL: @variadic_get(
kgen.func @variadic_get() -> i32 {
  // CHECK-NEXT: kgen.param.constant: i32 = <11>
  %0 = kgen.param.constant: !kgen.variadic<i32> = <[7, 11, 13]>
  %1 = index.constant 1
  %2 = pop.variadic.get %0[%1] : !kgen.variadic<i32>
  kgen.return %2 : i32
}

// CHECK-LABEL: @variadic_get_splat(
kgen.func @variadic_get_splat(%arg0: i32, %idx: index) -> i32 {
  // CHECK-NEXT: kgen.return %arg0 : i32
  %1 = pop.variadic.splat 4, %arg0 : !kgen.variadic<i32>
  %2 = pop.variadic.get %1[%idx] : !kgen.variadic<i32>
  kgen.return %2 : i32
}


// CHECK-LABEL: @variadic_create_get(
kgen.func @variadic_create_get(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-NEXT: kgen.return %arg0
  %0 = pop.variadic.create [%arg0, %arg1] : !kgen.variadic<i32>
  %1 = index.constant 0
  %2 = pop.variadic.get %0[%1] : !kgen.variadic<i32>
  kgen.return %2 : i32
}

// CHECK-LABEL: @variadic_size(
kgen.func @variadic_size() -> index {
  // CHECK-NEXT: kgen.param.constant = <3>
  %0 = kgen.param.constant: !kgen.variadic<i32> = <[7, 11, 13]>
  %1 = pop.variadic.size %0 : !kgen.variadic<i32>
  kgen.return %1 : index
}

// CHECK-LABEL: @variadic_create_size(
kgen.func @variadic_create_size(%arg0: i32, %arg1: i32) -> index {
  // CHECK-NEXT: kgen.param.constant = <2>
  %0 = pop.variadic.create [%arg0, %arg1] : !kgen.variadic<i32>
  %1 = pop.variadic.size %0 : !kgen.variadic<i32>
  kgen.return %1 : index
}

// CHECK-LABEL: @variadic_splat_size(
kgen.func @variadic_splat_size(%arg0: i32) -> index {
  // CHECK-NEXT: kgen.param.constant = <4>
  %0 = pop.variadic.splat 4, %arg0 : !kgen.variadic<i32>
  %1 = pop.variadic.size %0 : !kgen.variadic<i32>
  kgen.return %1 : index
}

// CHECK-LABEL: @dtype_to_ui8(
kgen.func @dtype_to_ui8() -> ui8 {
  // CHECK-NEXT: kgen.param.constant: ui8 = <1>
  %0 = kgen.param.constant: dtype = <bool>
  %1 = pop.dtype.to_ui8 %0
  kgen.return %1 : ui8
}

// CHECK-LABEL: @dtype_from_ui8(
kgen.func @dtype_from_ui8() -> !kgen.dtype {
  // CHECK-NEXT: kgen.param.constant: dtype = <bool>
  %0 = kgen.param.constant: ui8 = <1>
  %1 = pop.dtype.from_ui8 %0
  kgen.return %1 : !kgen.dtype
}

// CHECK-LABEL: @fold_offset
// CHECK: (%[[ARG0:.*]]:
kgen.func @fold_offset(%arg0: !kgen.pointer<index>) -> (!kgen.pointer<index>) {
  // CHECK-NEXT: kgen.return %[[ARG0]]
  %0 = kgen.param.constant = <0>
  %1 = pop.offset %arg0[%0] : !kgen.pointer<index>
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @select
kgen.func @select(%arg0: i1, %arg1: i32, %arg2: i32) -> (i32, i32) {
  // CHECK-NEXT: kgen.return %arg1, %arg2
  %true = kgen.param.constant: i1 = <1>
  %0 = pop.select %arg0, %arg1, %arg1 : i32
  %1 = pop.select %true, %arg2, %arg1 : i32
  kgen.return %0, %1 : i32, i32
}

// CHECK-LABEL: @select_to_cond
kgen.func @select_to_cond(%cond: i1) -> !pop.scalar<bool> {
  // CHECK-NEXT: %0 = pop.cast_from_builtin %arg0 : i1 to !pop.scalar<bool>
  // CHECK-NEXT: kgen.return %0
  %true = kgen.param.constant: scalar<bool> = <true>
  %false = kgen.param.constant: scalar<bool> = <false>
  %0 = pop.select %cond, %true, %false : !pop.scalar<bool>
  kgen.return %0: !pop.scalar<bool>
}


// CHECK-LABEL: @string_ops
kgen.func @string_ops() -> index {
  %str = kgen.param.constant: string = <"four">
  // CHECK-DAG: kgen.param.constant = <4>
  %0 = pop.string.size %str
  %hello_world = kgen.param.constant: string = <"hello world">
  %world = kgen.param.constant: string = <" world">
  %empty_str = kgen.param.constant: string = <"">
  kgen.return %0 : index
}


// CHECK-LABEL: @offset_of_offset
kgen.func @offset_of_offset(%arg0: !kgen.pointer<index>) -> !kgen.pointer<index> {
  %idx3 = index.constant 3
  %0 = pop.offset %arg0[%idx3] : !kgen.pointer<index>
  %idx200 = index.constant 200
  // CHECK: %0 = pop.offset %arg0[%index203]
  %1 = pop.offset %0[%idx200] : !kgen.pointer<index>
  // CHECK: return %0
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @large_int_memory_leak
// COM: Ensure that memory is correctly freed from a SIMDAttr.
kgen.func @large_int_memory_leak() -> !pop.scalar<si128> {
  // CHECK: constant: scalar<si128> = <1234>
  %0 = kgen.param.constant: si128 = <1234>
  %1 = pop.cast_from_builtin %0 : si128 to !pop.scalar<si128>
  kgen.return %1 : !pop.scalar<si128>
}

// CHECK-LABEL: kgen.func @select_true_false
kgen.func @select_true_false(%arg0: i1) -> i1 {
  // CHECK-NEXT: return %arg0 : i1
  %0 = kgen.param.constant: i1 = <1>
  %1 = kgen.param.constant: i1 = <0>
  %2 = pop.select %arg0, %0, %1 : i1
  kgen.return %2 : i1
}

// CHECK-LABEL: kgen.func @select_of_select
kgen.func @select_of_select(%arg0: i1, %arg1: index, %arg2: index, %arg3: index) -> (index, index) {
  // CHECK-NEXT: %0 = pop.select %arg0, %arg1, %arg3
  %0 = pop.select %arg0, %arg1, %arg2 : index
  %1 = pop.select %arg0, %0, %arg3 : index
  // CHECK-NEXT: %1 = pop.select %arg0, %arg1, %arg3
  %2 = pop.select %arg0, %arg2, %arg3 : index
  %3 = pop.select %arg0, %arg1, %2 : index
  // CHECK-NEXT: return %0, %1
  kgen.return %1, %3 : index, index
}

// CHECK-LABEL: kgen.func @lifetime_markers
kgen.func @lifetime_markers() {
  pop.stack_alloc.lifetime.start()
  pop.stack_alloc.lifetime.end()
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @load_bitcast
kgen.func @load_bitcast(%arg0: !kgen.pointer<pointer<none>>) -> !kgen.pointer<index> {
  // CHECK-NEXT: %0 = pop.load %arg0 : !kgen.pointer<pointer<none>>
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: %1 = pop.pointer.bitcast %0 : !kgen.pointer<none> to !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: return %1
  kgen.return %1 : !kgen.pointer<index>
}

// CHECK-LABEL: @load_bitcast_ptr_ptr
kgen.func @load_bitcast_ptr_ptr(%arg0: !kgen.pointer<none>) -> !kgen.pointer<none> {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<pointer<none>>
  // CHECK-NEXT: pop.load %0
  %1 = pop.load %0 : !kgen.pointer<pointer<none>>
  kgen.return %1 : !kgen.pointer<none>
}

// CHECK-LABEL: @load_bitcast_func_ptr
kgen.func @load_bitcast_func_ptr(%arg0: !kgen.generator<() -> ()>) ->
                                  (!kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index>) {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0
  %0 = pop.pointer.bitcast %arg0 : !kgen.generator<() -> ()> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.load %0
  %1 = pop.load %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.load volatile %0
  %2 = pop.load volatile %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.load volatile invariant %0
  %3 = pop.load volatile invariant %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.load atomic acquire %0
  %4 = pop.load atomic acquire %0 : !kgen.pointer<pointer<index>>
  kgen.return %1, %2, %3, %4 : !kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index>, !kgen.pointer<index>
}

// CHECK-LABEL: @load_of_store
kgen.func @load_of_store(%arg0: !kgen.pointer<index>, %arg1: index) -> index {
  // CHECK-NEXT: pop.store %arg1, %arg0
  pop.store %arg1, %arg0 : !kgen.pointer<index>
  %1 = pop.load %arg0 : !kgen.pointer<index>
  // CHECK-NEXT: return %arg1
  kgen.return %1 : index
}

// CHECK-LABEL: @load_of_store_atomic
// CHECK-SAME: [[PTR:%[a-z0-9]*]]:
// CHECK-SAME: [[VAL:%[a-z0-9]*]]:
kgen.func @load_of_store_atomic(%arg0: !kgen.pointer<index>, %arg1: index) -> index {
  // CHECK-NEXT: pop.store atomic release [[VAL]], [[PTR]]
  pop.store atomic release %arg1, %arg0 : !kgen.pointer<index>
  // CHECK-NEXT: pop.load atomic acquire [[PTR]]
  %1 = pop.load atomic acquire %arg0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: @atomic_store_not_canonicalized
// CHECK-SAME: [[PTR:%[a-z0-9]*]]:
// CHECK-SAME: [[VAL:%[a-z0-9]*]]:
kgen.func @atomic_store_not_canonicalized(%p: !kgen.pointer<scalar<f32>>, %v: !pop.scalar<f32>) {
  // CHECK: pop.store atomic release [[VAL]], [[PTR]]
  pop.store atomic release %v, %p : !kgen.pointer<scalar<f32>>
  // CHECK: pop.store atomic release [[VAL]], [[PTR]]
  pop.store atomic release %v, %p : !kgen.pointer<scalar<f32>>
  kgen.return
}

// CHECK-LABEL: @store_unknown
kgen.func @store_unknown(%ptr : !kgen.pointer<array<4, index>>) {
  %array = kgen.param.constant: array<4, index> = <#interp.uninitmem>
  pop.store %array, %ptr align<1> : !kgen.pointer<array<4, index>>

  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: @store_bitcast
kgen.func @store_bitcast(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<pointer<none>>) {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<none>
  %0 = pop.pointer.bitcast %arg1 : !kgen.pointer<pointer<none>> to !kgen.pointer<pointer<index>>
  // CHECK-NEXT: pop.store %0, %arg1 : !kgen.pointer<pointer<none>>
  pop.store %arg0, %0 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: %1 = pop.pointer.bitcast %arg0 : !kgen.pointer<index> to !kgen.pointer<none>
  // CHECK-NEXT: pop.store volatile %1, %arg1 : !kgen.pointer<pointer<none>>
  pop.store volatile %arg0, %0 : !kgen.pointer<pointer<index>>
  kgen.return
}

// CHECK-LABEL: @bitcast_free
kgen.func @bitcast_free(%arg0: !kgen.pointer<none>) {
  %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<index>
  // CHECK-NEXT: pop.aligned_free %arg0
  pop.aligned_free %0 : <index>
  kgen.return
}

// CHECK-LABEL: @variant_bitcast
kgen.func @variant_bitcast() -> !kgen.pointer<i32> {
  // CHECK: constant: pointer<i32> = <0>
  %0 = kgen.param.constant: pointer<variant<i32, i64>> = <0>
  %1 = pop.variant.bitcast %0, <0> : <variant<i32, i64>> as <i32>
  kgen.return %1 : !kgen.pointer<i32>
}

// CHECK-LABEL: @union_bitcast
kgen.func @union_bitcast() -> !kgen.pointer<i32> {
  %0 = kgen.param.constant: pointer<union<i32>> = <0>
  // CHECK-NEXT: constant: pointer<i32> = <0>
  %1 = pop.union.bitcast %0 : <union<i32>> as <i32>
  kgen.return %1 : !kgen.pointer<i32>
}

// CHECK-LABEL: @union_wrap
kgen.func @union_wrap() -> !pop.union<i32> {
  %0 = kgen.param.constant: i32 = <42>
  // CHECK-NEXT: constant: union<i32> = <{:i32 42}>
  %1 = pop.union.wrap %0 : i32 as <i32>
  kgen.return %1 : !pop.union<i32>
}

// CHECK-LABEL: @union_unwrap
kgen.func @union_unwrap() -> i32 {
  %0 = kgen.param.constant: union<i32> = <{:i32 42}>
  // CHECK-NEXT: constant: i32 = <42>
  %1 = pop.union.unwrap %0 : <i32> as i32
  kgen.return %1 : i32
}

kgen.func @union_unwrap_type() -> i64 {
  %0 = kgen.param.constant: union<i32, i64> = <{:i32 42}>
  // CHECK: pop.union.unwrap
  %1 = pop.union.unwrap %0 : <i32, i64> as i64
  kgen.return %1 : i64
}

// CHECK-LABEL: @wrap_unwrap
kgen.func @wrap_unwrap(%arg0: !pop.union<i32, i64>) -> !pop.union<i32, i64> {
  %0 = pop.union.unwrap %arg0 : <i32, i64> as i64
  %1 = pop.union.wrap %0 : i64 as <i32, i64>
  // CHECK-NEXT: return %arg0
  kgen.return %1 : !pop.union<i32, i64>
}

// CHECK-LABEL: @wrap_unwrap_type
kgen.func @wrap_unwrap_type(%arg0: !pop.union<i32>) -> !pop.union<i32, i64> {
  // CHECK-NEXT: pop.union.unwrap
  %0 = pop.union.unwrap %arg0 : <i32> as i32
  %1 = pop.union.wrap %0 : i32 as <i32, i64>
  kgen.return %1 : !pop.union<i32, i64>
}

// CHECK-LABEL: @unwrap_wrap
kgen.func @unwrap_wrap(%arg0: i32) -> i32 {
  %0 = pop.union.wrap %arg0 : i32 as <i32, i64>
  %1 = pop.union.unwrap %0 : <i32, i64> as i32
  // CHECK-NEXT: return %arg0
  kgen.return %1 : i32
}

// CHECK-LABEL: @unwrap_wrap_type
kgen.func @unwrap_wrap_type(%arg0: i32) -> i64 {
  // CHECK-NEXT: pop.union.wrap
  %0 = pop.union.wrap %arg0 : i32 as <i32, i64>
  %1 = pop.union.unwrap %0 : <i32, i64> as  i64
  kgen.return %1 : i64
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:32:32">} {
  // CHECK-LABEL: @cast_const_ui8_to_index_folding
  kgen.func @cast_const_ui8_to_index_folding() -> !pop.scalar<index> {
    // CHECK-NEXT: kgen.param.constant: scalar<index> = <128>
    %in = kgen.param.constant: scalar<ui8> = <128>
    %out = pop.cast %in : !pop.scalar<ui8> to !pop.scalar<index>
    kgen.return %out : !pop.scalar<index>
  }

  // CHECK-LABEL: @cast_si64_ui64_index
  kgen.func @cast_si64_ui64_index(%in : !pop.scalar<si64>) -> !pop.scalar<index> {
    // CHECK-NEXT: pop.cast {{.*}} : !pop.scalar<si64> to !pop.scalar<index>
    %1 = pop.cast %in : !pop.scalar<si64> to !pop.scalar<ui64>
    %2 = pop.cast %1 : !pop.scalar<ui64> to !pop.scalar<index>

    kgen.return %2 : !pop.scalar<index>
  }

  // CHECK-LABEL: @cast_si64_index_f64
  kgen.func @cast_si64_index_f64(%in : !pop.scalar<si64>) -> !pop.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<si64> to !pop.scalar<index>
    %1 = pop.cast %in : !pop.scalar<si64> to !pop.scalar<index>
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<index> to !pop.scalar<f64>
    %2 = pop.cast %1 : !pop.scalar<index> to !pop.scalar<f64>

    kgen.return %2 : !pop.scalar<f64>
  }

  // CHECK-LABEL: @cast_si32_index_f64
  kgen.func @cast_si32_index_f64(%in : !pop.scalar<si32>) -> !pop.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<si32> to !pop.scalar<index>
    %1 = pop.cast %in : !pop.scalar<si32> to !pop.scalar<index>
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<index> to !pop.scalar<f64>
    %2 = pop.cast %1 : !pop.scalar<index> to !pop.scalar<f64>

    kgen.return %2 : !pop.scalar<f64>
  }

  // CHECK-LABEL: @cast_index_ui32_f64
  kgen.func @cast_index_ui32_f64(%in : !pop.scalar<index>) -> !pop.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<index> to !pop.scalar<ui32>
    %1 = pop.cast %in : !pop.scalar<index> to !pop.scalar<ui32>
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<ui32> to !pop.scalar<f64>
    %2 = pop.cast %1 : !pop.scalar<ui32> to !pop.scalar<f64>

    kgen.return %2 : !pop.scalar<f64>
  }

  // CHECK-LABEL: @bitcast_index
  kgen.func @bitcast_index() -> (!pop.simd<2, ui32>, !pop.simd<2, index>,
                                 !pop.simd<2, index>, !pop.simd<2, f32>) {
    %ui32_const = kgen.param.constant: simd<2, ui32> = <<1, 2>>
    %f32_const = kgen.param.constant: simd<2, f32> = <<"1.5", "2.5">>
    %index_const = kgen.param.constant: simd<2, index> = <<3, 4>>

    %2 = pop.bitcast %index_const : !pop.simd<2, index> to !pop.simd<2, ui32>
    %3 = pop.bitcast %ui32_const : !pop.simd<2, ui32> to !pop.simd<2, index>
    %4 = pop.bitcast %f32_const : !pop.simd<2, f32> to !pop.simd<2, index>
    %5 = pop.bitcast %index_const : !pop.simd<2, index> to !pop.simd<2, f32>
    // CHECK: [[SIMD_UI32:%.*]] = kgen.param.constant: simd<2, ui32> = <<3, 4>>
    // CHECK-NEXT: [[SIMD_INDEX:%.*]] = kgen.param.constant: simd<2, index> = <<1, 2>>
    // CHECK-NEXT: [[SIMD_INDEX_F32:%.*]] = kgen.param.constant: simd<2, index> = <<1069547520, 1075838976>>
    // CHECK-NEXT: [[SIMD_F32:%.*]] = kgen.param.constant: simd<2, f32> = <<"4.20389539E-45", "5.60519386E-45">>
    // CHECK: kgen.return [[SIMD_UI32]], [[SIMD_INDEX]], [[SIMD_INDEX_F32]], [[SIMD_F32]]
    kgen.return %2, %3, %4, %5 : !pop.simd<2, ui32>, !pop.simd<2, index>,
                                 !pop.simd<2, index>, !pop.simd<2, f32>
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64">} {
  // CHECK-LABEL: @cast_const_ui8_to_index_folding
  kgen.func @cast_const_ui8_to_index_folding() -> !pop.scalar<index> {
    // CHECK-NEXT: kgen.param.constant: scalar<index> = <128>
    %in = kgen.param.constant: scalar<ui8> = <128>
    %out = pop.cast %in : !pop.scalar<ui8> to !pop.scalar<index>
    kgen.return %out : !pop.scalar<index>
  }

  // CHECK-LABEL: @cast_si64_ui64_index
  kgen.func @cast_si64_ui64_index(%in : !pop.scalar<si64>) -> !pop.scalar<index> {
    // CHECK-NEXT: pop.cast {{.*}} : !pop.scalar<si64> to !pop.scalar<index>
    %1 = pop.cast %in : !pop.scalar<si64> to !pop.scalar<ui64>
    %2 = pop.cast %1 : !pop.scalar<ui64> to !pop.scalar<index>

    kgen.return %2 : !pop.scalar<index>
  }

  // CHECK-LABEL: @cast_si64_index_f64
  kgen.func @cast_si64_index_f64(%in : !pop.scalar<si64>) -> !pop.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<si64> to !pop.scalar<index>
    %1 = pop.cast %in : !pop.scalar<si64> to !pop.scalar<index>
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<index> to !pop.scalar<f64>
    %2 = pop.cast %1 : !pop.scalar<index> to !pop.scalar<f64>

    kgen.return %2 : !pop.scalar<f64>
  }

  // CHECK-LABEL: @cast_si32_index_f64
  kgen.func @cast_si32_index_f64(%in : !pop.scalar<si32>) -> !pop.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<si32> to !pop.scalar<index>
    %1 = pop.cast %in : !pop.scalar<si32> to !pop.scalar<index>
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<index> to !pop.scalar<f64>
    %2 = pop.cast %1 : !pop.scalar<index> to !pop.scalar<f64>

    kgen.return %2 : !pop.scalar<f64>
  }

  // CHECK-LABEL: @cast_index_ui32_f64
  kgen.func @cast_index_ui32_f64(%in : !pop.scalar<index>) -> !pop.scalar<f64> {
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<index> to !pop.scalar<ui32>
    %1 = pop.cast %in : !pop.scalar<index> to !pop.scalar<ui32>
    // CHECK-NEXT:pop.cast {{.*}} : !pop.scalar<ui32> to !pop.scalar<f64>
    %2 = pop.cast %1 : !pop.scalar<ui32> to !pop.scalar<f64>

    kgen.return %2 : !pop.scalar<f64>
  }
}
