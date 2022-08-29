// This is the library of kernel generators used by the tests in this directory.
// It is also run as a test, which just verifies that it parses correctly.

// RUN: kgen-opt %s | kgen-opt -o /dev/null


//===----------------------------------------------------------------------===//
// div
//===----------------------------------------------------------------------===//

kgen.generator.interface @div<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @div_f32<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <[eq(:dtype type, f32), "incorrect element type"]> implements @div {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
  %2 = llvm.fdiv %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

kgen.generator @div_f64<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <[eq(:dtype type, f64), "incorrect element type"]> implements @div {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f64
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f64
  %2 = llvm.fdiv %0, %1 : f64
  %3 = meta.cast_from_builtin %2 : f64 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// buffer.loadOrValue
//===----------------------------------------------------------------------===//

kgen.generator.interface @buffer.loadOrValue<isLoad: i1, type: dtype>
  (%buffer: !meta.buffer<?, type>, %idx: index, %val: !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @buffer.loadOrValueImpl<isLoad: i1, type: dtype>
  (%buffer: !meta.buffer<?, type>, %idx: index, %val: !meta.scalar<type>) -> !meta.scalar<type>
  implements @buffer.loadOrValue {
  %isLoad = kgen.param.constant : i1 = <isLoad>
  %res = scf.if %isLoad -> !meta.scalar<type> {
    %t0 = pop.buffer.load %buffer[%idx] : !meta.buffer<?, type>
    scf.yield %t0 : !meta.scalar<type>
  } else {
    scf.yield %val : !meta.scalar<type>
  }
  kgen.return %res : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// horner evaluator
//===----------------------------------------------------------------------===//

kgen.generator.interface @horner<type: dtype>(%val: !meta.scalar<type>, %coefficients: !meta.buffer<?, type>) -> !meta.scalar<type>

// The horner(val, coeffs) where val is a scalar and coeffs is a list of
// coefficients  [c0, c1, c2, ..., cn] is defined by the following equation:
// horner(val, coeffs) = c0 + val * (c1 + val * (c2 + val * (... + val * cn)))
//                     = fma(val, horner(val, coeffs[1:]), c0)
kgen.generator @horner_impl<type: dtype>(%val: !meta.scalar<type>, %coefficients: !meta.buffer<?, type>) -> !meta.scalar<type>
  constraints <[in(:dtype type, [f32, f64]), "incorrect element type"]> implements @horner {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %zerofVal = pop.constant(0.0) : !meta.scalar<type>
  %zerof = meta.cast_to_builtin %zerofVal: !meta.scalar<type> to f32
  %numCoeffs = meta.buffer.size %coefficients: !meta.buffer<?, type>
  %res = scf.for %i = %zero to %numCoeffs step %one iter_args(%sum = %zerof) -> (f32) {
    %sumVal = meta.cast_from_builtin %sum : f32 to !meta.scalar<type>
    %coeff = pop.buffer.load %coefficients[%i] : !meta.buffer<?, type>
    %resVal = pop.fma %sumVal, %val, %coeff : !meta.scalar<type>
    %res = meta.cast_to_builtin %resVal: !meta.scalar<type> to f32
    scf.yield %res : f32
  }
  %result = meta.cast_from_builtin %res : f32 to !meta.scalar<type>
  kgen.return %result : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// exp
//===----------------------------------------------------------------------===//

kgen.generator.interface @exp<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>


// Compute exp using the llvm intrinsics.
kgen.generator @exp_intrinsic_f32<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <[eq(:dtype type, f32), "incorrect element type"]> implements @exp {
  %0 = meta.cast_to_builtin %x: !meta.scalar<type> to f32
  %1 = "llvm.intr.exp"(%0) : (f32) -> f32
  %2 = meta.cast_from_builtin %1 : f32 to !meta.scalar<type>
  kgen.return %2 : !meta.scalar<type>
}

kgen.generator @exp_intrinsic_f64<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <[eq(:dtype type, f64), "incorrect element type"]> implements @exp {
  %0 = meta.cast_to_builtin %x: !meta.scalar<type> to f64
  %1 = "llvm.intr.exp"(%0) : (f64) -> f64
  %2 = meta.cast_from_builtin %1 : f64 to !meta.scalar<type>
  kgen.return %2 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// index2D
//===----------------------------------------------------------------------===//
// Computes the 1D index from a 2D index (i, j) and the stride of the 2D array.
//===----------------------------------------------------------------------===//

kgen.generator.interface @index2D(%row: index, %col: index, %stride: index) -> index

kgen.generator @index2DImpl(%row: index, %col: index, %stride: index) -> index
  implements @index2D {
  %0 = arith.muli %row, %stride : index
  %1 = arith.addi %0, %col : index
  kgen.return %1 : index
}
