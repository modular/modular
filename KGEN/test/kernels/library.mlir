// This is the library of kernel generators used by the tests in this directory.
// It is also run as a test, which just verifies that it parses correctly.

// RUN: kgen-opt %s | kgen-opt -o /dev/null

//===----------------------------------------------------------------------===//
// buffer.loadOrValue
//===----------------------------------------------------------------------===//

kgen.generator.interface @buffer.loadOrValue<isLoad: i1, type: dtype>
  (%ptr: !pop.pointer<scalar<type>>, %idx: index, %val: !pop.scalar<type>) -> !pop.scalar<type>

kgen.generator @buffer.loadOrValueImpl<isLoad: i1, type: dtype>
  (%ptr: !pop.pointer<scalar<type>>, %idx: index, %val: !pop.scalar<type>) -> !pop.scalar<type>
  implements @buffer.loadOrValue {
  %zero = index.constant 0
  %isLoad = kgen.param.constant: i1 = <isLoad>
  %res = scf.if %isLoad -> !pop.scalar<type> {
    %ptrOffset = pop.offset %ptr[%idx] : !pop.pointer<scalar<type>>
    %t = pop.load %ptrOffset : !pop.pointer<scalar<type>>
    scf.yield %t : !pop.scalar<type>
  } else {
    scf.yield %val : !pop.scalar<type>
  }
  kgen.return %res : !pop.scalar<type>
}

//===----------------------------------------------------------------------===//
// polynomial_evaluate
//===----------------------------------------------------------------------===//

kgen.generator.interface @polynomial_evaluate<type: dtype, size>(
    %val: !pop.scalar<type>, %coefficients: !pop.pointer<scalar<type>>) -> !pop.scalar<type>

/// Evaluates a polynomial using the horner scheme.
///
/// The horner(val, coeffs) where val is a scalar and coeffs is a list of
/// coefficients [c0, c1, c2, ..., cn] is defined by the following equation:
/// horner(val, coeffs) = c0 + val * (c1 + val * (c2 + val * (... + val * cn)))
///                     = fma(val, horner(val, coeffs[1:]), c0)
kgen.generator @horner<type: dtype, size>(
    %val: !pop.scalar<type>, %coefficients: !pop.pointer<scalar<type>>) -> !pop.scalar<type>
  constraints <[in(:dtype type, [f32, f64]), "incorrect element type"]> implements @polynomial_evaluate {
  %zero = index.constant 0
  %one = index.constant 1
  %zero_si64 = kgen.param.constant: !pop.scalar<si64> = <#pop.simd<0>>
  %zerof = pop.cast %zero_si64 : !pop.scalar<si64> to !pop.scalar<type>
  %numCoeffs = kgen.param.constant = <size>
  %result = scf.for %i = %zero to %numCoeffs step %one iter_args(%sum = %zerof) -> !pop.scalar<type> {
    %ptr = pop.offset %coefficients[%i] : !pop.pointer<scalar<type>>
    %coeff = pop.load %ptr : !pop.pointer<scalar<type>>
    %res = pop.fma %sum, %val, %coeff : !pop.scalar<type>
    scf.yield %res : !pop.scalar<type>
  }
  kgen.return %result : !pop.scalar<type>
}

//===----------------------------------------------------------------------===//
// exp
//===----------------------------------------------------------------------===//

kgen.generator.interface @exp<type: dtype>(%x: !pop.scalar<type>) -> !pop.scalar<type>


// Compute exp using the llvm intrinsics.
kgen.generator @exp_intrinsic_f32<type: dtype>(%x: !pop.scalar<type>) -> !pop.scalar<type>
  constraints <[eq(:dtype type, f32), "incorrect element type"]> implements @exp {
  %0 = pop.cast_to_builtin %x: !pop.scalar<type> to f32
  %1 = "llvm.intr.exp"(%0) : (f32) -> f32
  %2 = pop.cast_from_builtin %1 : f32 to !pop.scalar<type>
  kgen.return %2 : !pop.scalar<type>
}

kgen.generator @exp_intrinsic_f64<type: dtype>(%x: !pop.scalar<type>) -> !pop.scalar<type>
  constraints <[eq(:dtype type, f64), "incorrect element type"]> implements @exp {
  %0 = pop.cast_to_builtin %x: !pop.scalar<type> to f64
  %1 = "llvm.intr.exp"(%0) : (f64) -> f64
  %2 = pop.cast_from_builtin %1 : f64 to !pop.scalar<type>
  kgen.return %2 : !pop.scalar<type>
}

//===----------------------------------------------------------------------===//
// index2D
//===----------------------------------------------------------------------===//
// Computes the 1D index from a 2D index (i, j) and the stride of the 2D array.
//===----------------------------------------------------------------------===//

kgen.generator.interface @index2D(%row: index, %col: index, %stride: index) -> index

kgen.generator @index2DImpl(%row: index, %col: index, %stride: index) -> index
  implements @index2D {
  %0 = index.mul %row, %stride
  %1 = index.add %0, %col
  kgen.return %1 : index
}
