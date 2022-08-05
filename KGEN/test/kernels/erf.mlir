// RUN: kgen-elaborate %s -library=%S/library.mlir | FileCheck %s

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator.interface @fma<type: dtype>(!meta.scalar<type>, !meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @buffer.load<size, type: dtype>(%buffer: !meta.buffer<size, type>, %idx: index) -> !meta.scalar<type>
kgen.generator.interface @buffer.store<size, type: dtype>(%value: !meta.scalar<type>, %buffer: !meta.buffer<size, type>, %idx: index) -> ()

kgen.generator.interface @erf_scalar<type: dtype>(%in: !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @erf_scalar_taylor<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <in_dtype(type, [f32, f64])> implements @erf_scalar {
  // Compute erf(x) = (2.0*x)/Sqrt(Pi) - (2*x^3)/(3.0*Sqrt(Pi)) in Horner form as
  // = x * (- 0.37612638903183752463 * x^2 + 1.1283791670955125739)
  // = x * fma(x^2, -0.37612638903183752463, 1.1283791670955125739)
  %c0 = pop.constant(1.1283791670955125739) : !meta.scalar<type>
  %c1 = pop.constant(-0.37612638903183752463) : !meta.scalar<type>
  %x2 = pop.mul %x, %x : !meta.scalar<type>
  %t0 = kgen.call @fma<type : dtype = type>(%x2, %c1, %c0) : (!meta.scalar<type>, !meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %t1 = pop.mul %t0, %x : !meta.scalar<type>
  kgen.return %t1 : !meta.scalar<type>
}

// CHECK-LABEL: kgen.kernel @"erf_scalar_taylor,type=f32"(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
// CHECK-LABEL: kgen.kernel @"erf_scalar_taylor,type=f64"(%arg0: !meta.scalar<f64>) -> !meta.scalar<f64> {

// Instantiate erf_scalar for f32 and f64.

// CHECK-LABEL: kgen.kernel @erf_scalar_f32(%arg0: f32) -> f32
// CHECK: kgen.call @"erf_scalar_taylor,type=f32"(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
kgen.kernel @erf_scalar_f32(%arg0: f32) -> f32 {
  %0 = meta.cast_from_builtin %arg0 : f32 to !meta.scalar<f32>
  %1 = kgen.call @erf_scalar<type: dtype = f32>(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f32> to f32
  kgen.return %2 : f32
}

// CHECK-LABEL: kgen.kernel @erf_scalar_f64(%arg0: f64) -> f64
// CHECK: kgen.call @"erf_scalar_taylor,type=f64"(%0) : (!meta.scalar<f64>) -> !meta.scalar<f64>
kgen.kernel @erf_scalar_f64(%arg0: f64) -> f64 {
  %0 = meta.cast_from_builtin %arg0 : f64 to !meta.scalar<f64>
  %1 = kgen.call @erf_scalar<type: dtype = f64>(%0) : (!meta.scalar<f64>) -> !meta.scalar<f64>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f64> to f64
  kgen.return %2 : f64
}

kgen.generator.interface @erf<N, type: dtype>(%in: !meta.buffer<N, type>, %out : !meta.buffer<N, type>)

kgen.generator @erf_impl1<N, type: dtype>(%in: !meta.buffer<N, type>, %out : !meta.buffer<N, type>)
  implements @erf {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %size = kgen.param.value = <N>

    scf.for %i = %zero to %size step %one {
        %src  = kgen.call @buffer.load<size = N, type:dtype = type>(%in, %i) : (!meta.buffer<N, type>, index) -> !meta.scalar<type>
        %res  = kgen.call @erf_scalar<type: dtype = type>(%src) : (!meta.scalar<type>) -> !meta.scalar<type>
        kgen.call @buffer.store<size = N, type:dtype = type>(%res, %out, %i) : (!meta.scalar<type>, !meta.buffer<N, type>, index) -> ()
    }
    kgen.return
  }

// Instantiate @erf for concrete buffer size and element type

// CHECK-LABEL: kgen.kernel @erf_10_f32(%arg0: !meta.buffer<10, f32>, %arg1: !meta.buffer<10, f32>)
// CHECK: kgen.call @"erf_impl1,N=10,type=f32"(%arg0, %arg1) : (!meta.buffer<10, f32>, !meta.buffer<10, f32>) -> ()
kgen.kernel @erf_10_f32(%in: !meta.buffer<10, f32>, %out: !meta.buffer<10, f32>) {
  kgen.call @erf<N=10, type:dtype=f32>(%in, %out) : (!meta.buffer<10, f32>, !meta.buffer<10, f32>) -> ()
  kgen.return
}
