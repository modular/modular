// RUN: kgen-elaborate %s -library=%S/library.mlir | FileCheck %s

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator.interface @sub<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @mul<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @div<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @fma<type: dtype>(!meta.scalar<type>, !meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @float_constant<value: f64, type: dtype>() -> !meta.scalar<type>

kgen.generator @scalar_erf<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type> {
  // Compute erf(x) = (2.0*x)/Sqrt(Pi) - (2*x^3)/(3.0*Sqrt(Pi)) in Horner form as
  // = x * (- 0.37612638903183752463 * x^2 + 1.1283791670955125739)
  // = x * fma(x^2, -0.37612638903183752463, 1.1283791670955125739)
  %c0 = kgen.call @float_constant<value : f64 = 1.1283791670955125739, type : dtype = type>() : () -> !meta.scalar<type>
  %c1 = kgen.call @float_constant<value : f64 = -0.37612638903183752463, type : dtype = type>() : () -> !meta.scalar<type>
  %x2 = kgen.call @mul<type : dtype = type>(%x, %x) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %t0 = kgen.call @fma<type : dtype = type>(%x2, %c1, %c0) : (!meta.scalar<type>, !meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %t1 = kgen.call @mul<type : dtype = type>(%t0, %x) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  kgen.return %t1 : !meta.scalar<type>
}

// CHECK-LABEL: kgen.kernel @"scalar_erf,type=f32"(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
// CHECK-LABEL: kgen.kernel @"scalar_erf,type=f64"(%arg0: !meta.scalar<f64>) -> !meta.scalar<f64> {

// Instantiate erf for f32.

// CHECK-LABEL: kgen.kernel @erf_f32(%arg0: f32) -> f32
// CHECK: kgen.call @"scalar_erf,type=f32"(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
kgen.kernel @erf_f32(%arg0: f32) -> f32 {
  kgen.param.bind type : dtype = <f32>
  %0 = meta.cast_from_builtin %arg0 : f32 to !meta.scalar<type>
  %1 = kgen.call @scalar_erf<type: dtype = type>(%0) : (!meta.scalar<type>) -> !meta.scalar<type>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<type> to f32
  kgen.return %2 : f32
}

// CHECK-LABEL: kgen.kernel @erf_f64(%arg0: f64) -> f64
// CHECK: kgen.call @"scalar_erf,type=f64"(%0) : (!meta.scalar<f64>) -> !meta.scalar<f64>
kgen.kernel @erf_f64(%arg0: f64) -> f64 {
  kgen.param.bind type : dtype = <f64>
  %0 = meta.cast_from_builtin %arg0 : f64 to !meta.scalar<type>
  %1 = kgen.call @scalar_erf<type: dtype = type>(%0) : (!meta.scalar<type>) -> !meta.scalar<type>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<type> to f64
  kgen.return %2 : f64
}
