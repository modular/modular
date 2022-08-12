// RUN: kgen-opt %s -elaborate-kernels="search-path=%S" | FileCheck %s

kgen.include "library.mlir"

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator.interface @exp<type: dtype>(!meta.scalar<type>) -> !meta.scalar<type>
kgen.generator.interface @buffer.load<type: dtype>(%buffer: !meta.buffer<?, type>, %idx: index) -> !meta.scalar<type>
kgen.generator.interface @buffer.store<type: dtype>(%value: !meta.scalar<type>, %buffer: !meta.buffer<?, type>, %idx: index) -> ()

kgen.generator.interface @erf_scalar<type: dtype>(%in: !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @erf_scalar_taylor<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <in(:dtype type, [f32, f64]), "incorrect element type"> implements @erf_scalar {
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


// Uses the same Erf approximation found in the MLAS library.
kgen.generator @erf_scalar_mlas<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <in(:dtype type, [f32, f64]), "incorrect element type"> implements @erf_scalar {
  %xAbs = pop.abs %x : !meta.scalar<type>
  %branchCut = pop.constant(0.921875) : !meta.scalar<type>
  // computes xAbs < branchCut
  %branch0 = pop.cmp lt, %xAbs, %branchCut : !meta.scalar<type>
  %branch = meta.cast_to_builtin %branch0: !meta.scalar<bool> to i1
  %res = scf.if %branch -> !meta.scalar<type> {
    %c0 = pop.constant(1.72948930e-5 : f32) : !meta.scalar<type>
    %c1 = pop.constant(-3.83208680e-4 : f32) : !meta.scalar<type>
    %c2 = pop.constant(3.88393435e-3 : f32) : !meta.scalar<type>
    %c3 = pop.constant(-2.42545605e-2 : f32) : !meta.scalar<type>
    %c4 = pop.constant(1.06777847e-1 : f32) : !meta.scalar<type>
    %c5 = pop.constant(6.34846687e-1 : f32) : !meta.scalar<type>
    %c6 = pop.constant(1.28717512e-1 : f32) : !meta.scalar<type>
    %t0 = pop.fma %xAbs, %c1, %c6 : !meta.scalar<type>
    %t1 = pop.fma %xAbs, %t0, %c5 : !meta.scalar<type>
    %t2 = pop.fma %xAbs, %t1, %c4 : !meta.scalar<type>
    %t3 = pop.fma %xAbs, %t2, %c3 : !meta.scalar<type>
    %t4 = pop.fma %xAbs, %t3, %c2 : !meta.scalar<type>
    %t5 = pop.fma %xAbs, %t4, %c1 : !meta.scalar<type>
    %t6 = pop.fma %xAbs, %t5, %c0 : !meta.scalar<type>
    %t7 = pop.fma %t6, %xAbs, %xAbs : !meta.scalar<type>
    %t8 = pop.neg %t7 : !meta.scalar<type>
    %t9 = kgen.call @exp<type : dtype = type>(%t8) : (!meta.scalar<type>) -> !meta.scalar<type>
    %one = pop.constant(1) : !meta.scalar<type>
    %t10 = pop.sub %one, %t9 : !meta.scalar<type>
    %t11 = pop.copysign %t10, %x : !meta.scalar<type>
    scf.yield %t11 : !meta.scalar<type>
  } else {
    %c0 = pop.constant(-5.99104969e-4 : f32) : !meta.scalar<type>
    %c1 = pop.constant(4.99339588e-3 : f32) : !meta.scalar<type>
    %c2 = pop.constant(-2.67667342e-2 : f32) : !meta.scalar<type>
    %c3 = pop.constant(1.12818025e-1 : f32) : !meta.scalar<type>
    %c4 = pop.constant(-3.76124859e-1 : f32) : !meta.scalar<type>
    %c5 = pop.constant(1.28379151e-1 : f32) : !meta.scalar<type>
    %xSquared = pop.mul %x, %x : !meta.scalar<type>
    %t0 = pop.fma %xSquared, %c1, %c5 : !meta.scalar<type>
    %t1 = pop.fma %xSquared, %t0, %c4 : !meta.scalar<type>
    %t2 = pop.fma %xSquared, %t1, %c3 : !meta.scalar<type>
    %t3 = pop.fma %xSquared, %t2, %c2 : !meta.scalar<type>
    %t4 = pop.fma %xSquared, %t3, %c1 : !meta.scalar<type>
    %t5 = pop.fma %xSquared, %t4, %c0 : !meta.scalar<type>
    %t6 = pop.fma %t5, %x, %x : !meta.scalar<type>
    scf.yield %t6 : !meta.scalar<type>
  }
  kgen.return %res : !meta.scalar<type>
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

kgen.generator.interface @erf<type: dtype>(%in: !meta.buffer<?, type>, %out : !meta.buffer<?, type>)

kgen.generator @erf_impl1<type: dtype>(%in: !meta.buffer<?, type>, %out : !meta.buffer<?, type>)
  implements @erf {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index

  // TODO: Must assert that size of in and out buffers are the same
  %size = meta.buffer.size %in: !meta.buffer<?, type>

  scf.for %i = %zero to %size step %one {
      %src  = kgen.call @buffer.load<type:dtype = type>(%in, %i) : (!meta.buffer<?, type>, index) -> !meta.scalar<type>
      %res  = kgen.call @erf_scalar<type: dtype = type>(%src) : (!meta.scalar<type>) -> !meta.scalar<type>
      kgen.call @buffer.store<type:dtype = type>(%res, %out, %i) : (!meta.scalar<type>, !meta.buffer<?, type>, index) -> ()
  }

  kgen.return
}

// Instantiate @erf for concrete buffer size and element type

// CHECK-LABEL: kgen.kernel @erf_f32(%arg0: !meta.buffer<?, f32>, %arg1: !meta.buffer<?, f32>)
// CHECK: kgen.call @"erf_impl1,type=f32"(%arg0, %arg1) : (!meta.buffer<?, f32>, !meta.buffer<?, f32>) -> ()
kgen.kernel @erf_f32(%in: !meta.buffer<?, f32>, %out: !meta.buffer<?, f32>) {
  kgen.call @erf<type: dtype = f32>(%in, %out) : (!meta.buffer<?, f32>, !meta.buffer<?, f32>) -> ()
  kgen.return
}
