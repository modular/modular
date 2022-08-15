// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: kgen.kernel @pop_constant() -> !meta.scalar<f32> {
kgen.kernel @pop_constant() -> !meta.scalar<f32> {
  // CHECK-NEXT: pop.constant(32 : si64) : !meta.scalar<si64>
  %0 = pop.constant(32 : si64) : !meta.scalar<si64>
  // CHECK-NEXT: pop.constant(3.200000e+01 : f32) : !meta.scalar<f32>
  %1 = pop.constant(32.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: pop.constant(3.200000e+01 : f64) : !meta.scalar<f32>
  %2 = pop.constant(32.0 : f64) : !meta.scalar<f32>
  // CHECK-NEXT: pop.constant(32 : i64) : !meta.scalar<f32>
  %3 = pop.constant(32) : !meta.scalar<f32>
  // CHECK-NEXT: pop.constant(32 : si64) : !meta.scalar<si32>
  %4 = pop.constant(32 : si64) : !meta.scalar<si32>
  kgen.return %1 : !meta.scalar<f32>
}

// CHECK-LABEL: @pop_constant_simd
kgen.kernel @pop_constant_simd() {
  // CHECK: pop.constant(dense<[32, 64]>
  %0 = pop.constant(dense<[32, 64]> : vector<2xsi64>) : !meta.simd<2, si32>
  // CHECK: pop.constant(dense<[32, 64]>
  %1 = pop.constant(dense<[32, 64]> : vector<2xi32>) : !meta.simd<2, f64>
  // CHECK: pop.constant(dense<[32, 64]>
  %2 = pop.constant(dense<[32, 64]> : vector<2xi32>) : !meta.simd<2, ui64>
  kgen.return
}

// CHECK-LABEL: kgen.generator @pop_constant2<type: dtype>() -> !meta.scalar<type> {
kgen.generator @pop_constant2<type: dtype>() -> !meta.scalar<type> {
  // CHECK-NEXT: pop.constant(32 : i64) : !meta.scalar<type>
  %0 = pop.constant(32) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// CHECK-LABEL: kgen.kernel @pop_abs(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_abs(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %0 = pop.abs %arg0 : !meta.scalar<f32>
  %0 = pop.abs %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_neg(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_neg(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %0 = pop.neg %arg0 : !meta.scalar<f32>
  %0 = pop.neg %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_add() -> !meta.scalar<f32> {
kgen.kernel @pop_add() -> !meta.scalar<f32> {
  // CHECK-NEXT: %[[CST:.*]] = pop.constant(4.000000e+00 : f32) : !meta.scalar<f32>
  %a = pop.constant(4.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %[[CST0:.*]] = pop.constant(6.000000e+00 : f32) : !meta.scalar<f32>
  %b = pop.constant(6.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %0 = pop.add %[[CST]], %[[CST0]] : !meta.scalar<f32>
  %c = pop.add %a, %b : !meta.scalar<f32>
  kgen.return %c : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.generator @pop_add2<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type> {
kgen.generator @pop_add2<type: dtype>(%a: !meta.scalar<type>, %b: !meta.scalar<type>) -> !meta.scalar<type> {
  // CHECK-NEXT: %0 = pop.add %arg0, %arg1 : !meta.scalar<type>
  %c = pop.add %a, %b : !meta.scalar<type>
  kgen.return %c : !meta.scalar<type>
}

// CHECK-LABEL: kgen.kernel @pop_add_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_add_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.add %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.add %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_sub(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_sub(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %0 = pop.sub %arg0, %arg1 : !meta.scalar<f32>
  %0 = pop.sub %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_sub_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_sub_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.sub %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.sub %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_mul(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_mul(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %0 = pop.mul %arg0, %arg1 : !meta.scalar<f32>
  %0 = pop.mul %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_mul_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_mul_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.mul %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.mul %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_copysign(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_copysign(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %0 = pop.copysign %arg0, %arg1 : !meta.scalar<f32>
  %0 = pop.copysign %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_copysign_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_copysign_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.copysign %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.copysign %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_fma(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_fma(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %0 = pop.fma %arg0, %arg1, %arg2 : !meta.scalar<f32>
  %0 = pop.fma %arg0, %arg1, %arg2: !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_fma_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>, %arg2: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_fma_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>, %arg2 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.fma %arg0, %arg1, %arg2 : !meta.simd<4, f32>
  %0 = pop.fma %arg0, %arg1, %arg2 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: @pop_cmp
kgen.kernel @pop_cmp(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<bool> {
  // CHECK: pop.cmp ge, %{{.*}}, %{{.*}} :
  %0 = pop.cmp ge, %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<bool>
}

kgen.kernel @pop_cmp_simd(
    %arg0: !meta.simd<4, si32>, %arg1: !meta.simd<4, si32>,
    %arg2: !meta.simd<2, f64>, %arg3: !meta.simd<2, f64>
  ) -> (!meta.simd<4, bool>, !meta.simd<2, bool>) {
  // CHECK: pop.cmp ne, %{{.*}}, %{{.*}} :
  %0 = pop.cmp ne, %arg0, %arg1 : !meta.simd<4, si32>
  // CHECK: pop.cmp lt, %{{.*}}, %{{.*}} :
  %1 = pop.cmp lt, %arg2, %arg3 : !meta.simd<2, f64>
  kgen.return %0, %1 : !meta.simd<4, bool>, !meta.simd<2, bool>
}


// CHECK-LABEL: @pop_select
kgen.kernel @pop_select(%arg0 : !meta.scalar<bool>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: pop.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: @pop_select_simd
kgen.kernel @pop_select_simd(
    %arg0: !meta.simd<4, bool>,
    %arg1: !meta.simd<4, si32>,
    %arg2: !meta.simd<4, si32>
  ) -> !meta.simd<4, si32> {
  // CHECK: pop.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.simd<4, si32>
  kgen.return %0 : !meta.simd<4, si32>
}

// COM: Compute erf(x) = (2.0*x)/Sqrt(Pi) - (2*x^3)/(3.0*Sqrt(Pi)) in Horner form as
// COM: = x * (- 0.37612638903183752463 * x^2 + 1.1283791670955125739)

// CHECK-LABEL: kgen.kernel @erf
// CHECK: %[[X:.*]]:
kgen.kernel @erf(%x: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %[[CST:.*]] = pop.constant(1.1283791670955099 : f64) : !meta.scalar<f32>
  %c0 = pop.constant(1.12837916709551) : !meta.scalar<f32>
  // CHECK: %[[CST0:.*]] = pop.constant(-0.37612638903180001 : f64) : !meta.scalar<f32>
  %c1 = pop.constant(-0.3761263890318) : !meta.scalar<f32>
  // CHECK: %[[V0:.*]] = pop.mul %[[X]], %[[X]] : !meta.scalar<f32>
  %x2 = pop.mul %x, %x : !meta.scalar<f32>
  // CHECK: %[[V1:.*]] = pop.mul %[[V0]], %[[CST0]] : !meta.scalar<f32>
  %x3 = pop.mul %x2, %c1 : !meta.scalar<f32>
  // CHECK: %[[V2:.*]] = pop.add %[[V1]], %[[CST]] : !meta.scalar<f32>
  %x4 = pop.add %x3, %c0 : !meta.scalar<f32>
  // CHECK: pop.mul %[[V2]], %[[X]] : !meta.scalar<f32>
  %x5 = pop.mul %x4, %x : !meta.scalar<f32>
  kgen.return %x5 : !meta.scalar<f32>
}
