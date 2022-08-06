// This is the library of kernel generators used by the tests in this directory.
// It is also run as a test, which just verifies that it parses correctly.

// RUN: kgen-opt %s | kgen-opt -o /dev/null


//===----------------------------------------------------------------------===//
// div
//===----------------------------------------------------------------------===//

kgen.generator.interface @div<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @div_f32<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32), "incorrect element type"> implements @div {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
  %2 = llvm.fdiv %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

kgen.generator @div_f64<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f64), "incorrect element type"> implements @div {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f64
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f64
  %2 = llvm.fdiv %0, %1 : f64
  %3 = meta.cast_from_builtin %2 : f64 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// buffer.load
//===----------------------------------------------------------------------===//

kgen.generator.interface @buffer.load<type: dtype>(%buffer: !meta.buffer<?, type>, %idx: index) -> !meta.scalar<type>

kgen.generator @buffer_load_f32<type: dtype>(%buffer: !meta.buffer<?, type>, %idx: index) -> !meta.scalar<type>
  implements @buffer.load {
  %ptr = meta.buffer.address %buffer: !meta.buffer<?, type>
  %llvm_ptr = builtin.unrealized_conversion_cast %ptr: !meta.pointer<type> to !llvm.ptr<f32>
  %i64_idx = builtin.unrealized_conversion_cast %idx: index to i64
  %element_ptr = llvm.getelementptr %llvm_ptr[%i64_idx]: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %scalar_value = llvm.load %element_ptr : !llvm.ptr<f32>
  %value = meta.cast_from_builtin %scalar_value: f32 to !meta.scalar<type>
  kgen.return %value : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// buffer.store
//===----------------------------------------------------------------------===//

kgen.generator.interface @buffer.store<type: dtype>(%value: !meta.scalar<type>, %buffer: !meta.buffer<?, type>, %idx: index) -> ()

// TODO: Currently, the signature of @buffer_store_f32 must exactly match the signature of @buffer_store, so we can't
// have the signature use f32. When we allow this, we should change the signature to be specialized to f32
kgen.generator @buffer_store_f32<type: dtype>(%value: !meta.scalar<type>, %buffer: !meta.buffer<?, type>, %idx: index) -> ()
  implements @buffer.store {
  %ptr = meta.buffer.address %buffer: !meta.buffer<?, type>
  %llvm_ptr = builtin.unrealized_conversion_cast %ptr: !meta.pointer<type> to !llvm.ptr<f32>
  %i64_idx = builtin.unrealized_conversion_cast %idx: index to i64
  %scalar_val = meta.cast_to_builtin %value: !meta.scalar<type> to f32
  %element_ptr = llvm.getelementptr %llvm_ptr[%i64_idx]: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  llvm.store %scalar_val, %element_ptr : !llvm.ptr<f32>
  kgen.return
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
  constraints <in_dtype(type, [f32, f64]), "incorrect element type"> implements @horner {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %zerofVal = pop.constant(0.0) : !meta.scalar<type>
  %zerof = meta.cast_to_builtin %zerofVal: !meta.scalar<type> to f32
  %numCoeffs = meta.buffer.size %coefficients: !meta.buffer<?, type>
  %res = scf.for %i = %zero to %numCoeffs step %one iter_args(%sum = %zerof) -> (f32) {
    %sumVal = meta.cast_from_builtin %sum : f32 to !meta.scalar<type>
    %coeff = kgen.call @buffer.load<type : dtype = type>(%coefficients, %i): (!meta.buffer<?, type>, index) -> !meta.scalar<type>
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
  constraints <eq_dtype(type, f32), "incorrect element type"> implements @exp {
  %0 = meta.cast_to_builtin %x: !meta.scalar<type> to f32
  %1 = "llvm.intr.exp"(%0) : (f32) -> f32
  %2 = meta.cast_from_builtin %1 : f32 to !meta.scalar<type>
  kgen.return %2 : !meta.scalar<type>
}

kgen.generator @exp_intrinsic_f64<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f64), "incorrect element type"> implements @exp {
  %0 = meta.cast_to_builtin %x: !meta.scalar<type> to f64
  %1 = "llvm.intr.exp"(%0) : (f64) -> f64
  %2 = meta.cast_from_builtin %1 : f64 to !meta.scalar<type>
  kgen.return %2 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// lessThan
//===----------------------------------------------------------------------===//
// TODO: This is a temporary placeholder until the pop dialect has support
// for compare operations.
//===----------------------------------------------------------------------===//

kgen.generator.interface @lessThan<type: dtype>(%a: !meta.scalar<type>, %b: !meta.scalar<type>) -> i1

kgen.generator @lessThan_f32<type: dtype>(%a: !meta.scalar<type>, %b: !meta.scalar<type>) -> i1
  constraints <eq_dtype(type, f32), "incorrect element type"> implements @lessThan {
  %0 = meta.cast_to_builtin %a: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %b: !meta.scalar<type> to f32
  %2 = arith.cmpf olt, %0, %1 : f32
  kgen.return %2 : i1
}
