// RUN: kgen-opt %s | FileCheck %s

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
