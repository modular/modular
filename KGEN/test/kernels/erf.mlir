// RUN: kgen-opt %s -lower-lit -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library.mlir"

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator.interface @exp<type: dtype>(!pop.scalar<type>) -> !pop.scalar<type>

kgen.generator.interface @erf_scalar<type: dtype>(%in: !pop.scalar<type>) -> !pop.scalar<type>

lit.func @erf_scalar_taylor<type: dtype>(%x: !pop.scalar<type>) -> !pop.scalar<type>
  constraints <[in(:dtype type, [f32, f64]), "incorrect element type"]> implements @erf_scalar {
  // Compute erf(x) = (2.0*x)/Sqrt(Pi) - (2*x^3)/(3.0*Sqrt(Pi)) in Horner form as
  // = x * (- 0.37612638903183752463 * x^2 + 1.1283791670955125739)
  // = x * fma(x^2, -0.37612638903183752463, 1.1283791670955125739)
  %c0 = pop.constant(1.1283791670955125739) : !pop.scalar<type>
  %c1 = pop.constant(-0.37612638903183752463) : !pop.scalar<type>
  %x2 = pop.mul %x, %x : !pop.scalar<type>
  %t0 = pop.fma %x2, %c1, %c0 : !pop.scalar<type>
  %t1 = pop.mul %t0, %x : !pop.scalar<type>
  kgen.return %t1 : !pop.scalar<type>
}

// CHECK-LABEL: kgen.func @"erf_scalar_taylor,type=f32"(%{{.*}}: !pop.scalar<f32>) -> !pop.scalar<f32> {
// CHECK-LABEL: kgen.func @"erf_scalar_taylor,type=f64"(%{{.*}}: !pop.scalar<f64>) -> !pop.scalar<f64> {

// Instantiate erf_scalar for f32 and f64.

// CHECK-LABEL: kgen.func @erf_scalar_f32
// CHECK-SAME: %[[ARG0:.*]]: f32
// CHECK: %[[V0:.*]] = pop.cast_from_builtin %[[ARG0]]
// CHECK: kgen.call @"erf_scalar_taylor,type=f32"(%[[V0]]) : (!pop.scalar<f32>) -> !pop.scalar<f32>
kgen.generator @erf_scalar_f32(%arg0: f32) -> f32 {
  %0 = pop.cast_from_builtin %arg0 : f32 to !pop.scalar<f32>
  %1 = kgen.call @erf_scalar<type: dtype = f32>(%0) : (!pop.scalar<f32>) -> !pop.scalar<f32>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<f32> to f32
  kgen.return %2 : f32
}

// CHECK-LABEL: kgen.func @erf_scalar_f64
// CHECK-SAME: %[[ARG0:.*]]: f64
// CHECK: %[[V0:.*]] = pop.cast_from_builtin %[[ARG0]]
// CHECK: kgen.call @"erf_scalar_taylor,type=f64"(%[[V0]]) : (!pop.scalar<f64>) -> !pop.scalar<f64>
kgen.generator @erf_scalar_f64(%arg0: f64) -> f64 {
  %0 = pop.cast_from_builtin %arg0 : f64 to !pop.scalar<f64>
  %1 = kgen.call @erf_scalar<type: dtype = f64>(%0) : (!pop.scalar<f64>) -> !pop.scalar<f64>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<f64> to f64
  kgen.return %2 : f64
}

kgen.generator.interface @erf<type: dtype>(%in: !zap.buffer<?, type>, %out : !zap.buffer<?, type>)

lit.func @erf_impl1<type: dtype>(%in: !zap.buffer<?, type>, %out : !zap.buffer<?, type>)
  implements @erf {
  %zero = index.constant 0
  %one = index.constant 1
  %undef = pop.constant(0) : !pop.scalar<type>
  %undefVec = pop.simd.splat %undef : !pop.simd<1, type>

  // TODO: Must assert that size of in and out buffers are the same
  %size = zap.buffer.size %in: !zap.buffer<?, type>

  scf.for %i = %zero to %size step %one {
      %src0  = zap.buffer.load %in[%i] : !zap.buffer<?, type>, !pop.simd<1, type>
      %src  = pop.simd.extractelement %src0[%zero] : !pop.simd<1, type>
      %res0  = kgen.call @erf_scalar<type: dtype = type>(%src) : (!pop.scalar<type>) -> !pop.scalar<type>
      %res  = pop.simd.insertelement %res0, %undefVec[%zero] : !pop.simd<1, type>
      zap.buffer.store %res, %out[%i] : !pop.simd<1, type>, !zap.buffer<?, type>
  }

  kgen.return
}

// Instantiate @erf for concrete buffer size and element type

// CHECK-LABEL: kgen.func @erf_f32
// CHECK-SAME: %[[ARG0:.*]]: !zap.buffer<?, f32>, %[[ARG1:.*]]: !zap.buffer<?, f32>
// CHECK: kgen.call @"erf_impl1,type=f32"(%[[ARG0]], %[[ARG1]]) : (!zap.buffer<?, f32>, !zap.buffer<?, f32>) -> ()
kgen.generator @erf_f32(%in: !zap.buffer<?, f32>, %out: !zap.buffer<?, f32>) {
  kgen.call @erf<type: dtype = f32>(%in, %out) : (!zap.buffer<?, f32>, !zap.buffer<?, f32>) -> ()
  kgen.return
}
