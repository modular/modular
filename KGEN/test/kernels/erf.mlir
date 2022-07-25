// RUN: kgen-elaborate %s -library=%S/library.mlir | FileCheck %s

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator.interface @sub<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @mul<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @div<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @float_constant<value: f64, type: dtype>() -> !meta.scalar<type>

kgen.generator @scalar_erf<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> {
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

// CHECK: kgen.kernel @dummy()
kgen.kernel @dummy() {
  kgen.return
}