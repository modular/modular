// RUN: kgen-opt %s -lower-hlkgen -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library.mlir"

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator.interface @exp<type: dtype>(!meta.scalar<type>) -> !meta.scalar<type>

kgen.generator.interface @erf_scalar<type: dtype>(%in: !meta.scalar<type>) -> !meta.scalar<type>

hlkgen.generator @erf_scalar_taylor<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <[in(:dtype type, [f32, f64]), "incorrect element type"]> implements @erf_scalar {
  // Compute erf(x) = (2.0*x)/Sqrt(Pi) - (2*x^3)/(3.0*Sqrt(Pi)) in Horner form as
  // = x * (- 0.37612638903183752463 * x^2 + 1.1283791670955125739)
  // = x * fma(x^2, -0.37612638903183752463, 1.1283791670955125739)
  %c0 = pop.constant(1.1283791670955125739) : !meta.scalar<type>
  %c1 = pop.constant(-0.37612638903183752463) : !meta.scalar<type>
  %x2 = pop.mul %x, %x : !meta.scalar<type>
  %t0 = pop.fma %x2, %c1, %c0 : !meta.scalar<type>
  %t1 = pop.mul %t0, %x : !meta.scalar<type>
  kgen.return %t1 : !meta.scalar<type>
}

// CHECK-LABEL: kgen.func @"erf_scalar_taylor,type=f32"(%{{.*}}: !meta.scalar<f32>) -> !meta.scalar<f32> {
// CHECK-LABEL: kgen.func @"erf_scalar_taylor,type=f64"(%{{.*}}: !meta.scalar<f64>) -> !meta.scalar<f64> {

// Instantiate erf_scalar for f32 and f64.

// CHECK-LABEL: kgen.func @erf_scalar_f32
// CHECK-SAME: %[[ARG0:.*]]: f32
// CHECK: %[[V0:.*]] = pop.type_raise %[[ARG0]]
// CHECK: kgen.call @"erf_scalar_taylor,type=f32"(%[[V0]]) : (!meta.scalar<f32>) -> !meta.scalar<f32>
kgen.generator @erf_scalar_f32(%arg0: f32) -> f32 {
  %0 = pop.type_raise %arg0 : f32 to !meta.scalar<f32>
  %1 = kgen.call @erf_scalar<type: dtype = f32>(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  %2 = pop.type_lower %1 : !meta.scalar<f32> to f32
  kgen.return %2 : f32
}

// CHECK-LABEL: kgen.func @erf_scalar_f64
// CHECK-SAME: %[[ARG0:.*]]: f64
// CHECK: %[[V0:.*]] = pop.type_raise %[[ARG0]]
// CHECK: kgen.call @"erf_scalar_taylor,type=f64"(%[[V0]]) : (!meta.scalar<f64>) -> !meta.scalar<f64>
kgen.generator @erf_scalar_f64(%arg0: f64) -> f64 {
  %0 = pop.type_raise %arg0 : f64 to !meta.scalar<f64>
  %1 = kgen.call @erf_scalar<type: dtype = f64>(%0) : (!meta.scalar<f64>) -> !meta.scalar<f64>
  %2 = pop.type_lower %1 : !meta.scalar<f64> to f64
  kgen.return %2 : f64
}

kgen.generator.interface @erf<type: dtype>(%in: !zap.buffer<?, type>, %out : !zap.buffer<?, type>)

hlkgen.generator @erf_impl1<type: dtype>(%in: !zap.buffer<?, type>, %out : !zap.buffer<?, type>)
  implements @erf {
  %zero = index.constant 0
  %one = index.constant 1

  // TODO: Must assert that size of in and out buffers are the same
  %size = zap.buffer.size %in: !zap.buffer<?, type>

  scf.for %i = %zero to %size step %one {
      %src  = zap.buffer.load %in[%i] : !zap.buffer<?, type>
      %res  = kgen.call @erf_scalar<type: dtype = type>(%src) : (!meta.scalar<type>) -> !meta.scalar<type>
      zap.buffer.store %res, %out[%i] : !zap.buffer<?, type>
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
