// RUN: kgen-elaborate %s -library=%S/library.mlir | FileCheck %s

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator.interface @sub<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @mul<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @div<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @float_constant<value: f64, type: dtype>() -> !meta.scalar<type>

kgen.generator @scalar_erf<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type> {
  // Compute 2/sqrt(pi) * (x - x^3 / 3) as 2/sqrt(pi) * x * (1 - x^2 / 3)
  %sqrt_of_pi = kgen.call @float_constant<value : f64 = 1.77245384, type : dtype = type>() : () -> !meta.scalar<type>
  %one     = kgen.call @float_constant<value: f64 = 1.0, type : dtype = type>() : () -> !meta.scalar<type>
  %two     = kgen.call @float_constant<value: f64 = 2.0, type : dtype = type>() : () -> !meta.scalar<type>
  %three   = kgen.call @float_constant<value: f64 = 3.0, type : dtype = type>() : () -> !meta.scalar<type>
  %fact1   = kgen.call @div<type: dtype = type>(%two, %sqrt_of_pi) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %x_sqr   = kgen.call @mul<type: dtype = type>(%x, %x) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %x_sqr_3 = kgen.call @div<type: dtype = type>(%x_sqr, %three) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %fact3   = kgen.call @sub<type: dtype = type>(%one, %x_sqr_3) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %prod1   = kgen.call @mul<type: dtype = type>(%fact1, %x) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %prod2   = kgen.call @mul<type: dtype = type>(%prod1, %fact3) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  kgen.return %prod2 : !meta.scalar<type>
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


