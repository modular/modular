// This is the library of kernel generators used by the tests in this directory.
// It is also run as a test, which just verifies that it parses correctly.

// RUN: kgen-opt %s | kgen-opt -o /dev/null

//===----------------------------------------------------------------------===//
// add
//===----------------------------------------------------------------------===//

kgen.generator.interface @add<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @add_i32<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  // TODO: This also works for ui32
  constraints <eq_dtype(type, si32)> implements @add {
    %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to i32
    %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to i32
    %2 = arith.addi %0, %1 : i32
    %3 = meta.cast_from_builtin %2 : i32 to !meta.scalar<type>
    kgen.return %3 : !meta.scalar<type>
  }

kgen.generator @add_i64<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  // TODO: This also works for ui64
  constraints <eq_dtype(type, si64)> implements @add {
    %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to i64
    %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to i64
    %2 = arith.addi %0, %1 : i64
    %3 = meta.cast_from_builtin %2 : i64 to !meta.scalar<type>
    kgen.return %3 : !meta.scalar<type>
  }

kgen.generator @add_f32<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @add {
    %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
    %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
    %2 = arith.addf %0, %1 : f32
    %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
    kgen.return %3 : !meta.scalar<type>
  }

kgen.generator @add_f64<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f64)> implements @add {
    %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f64
    %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f64
    %2 = arith.addf %0, %1 : f64
    %3 = meta.cast_from_builtin %2 : f64 to !meta.scalar<type>
    kgen.return %3 : !meta.scalar<type>
  }

//===----------------------------------------------------------------------===//
// sub
//===----------------------------------------------------------------------===//

// TODO: Add support for types other than f32.
kgen.generator.interface @sub<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @sub_f32<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @sub  {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
  %2 = llvm.fsub %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

kgen.generator @sub_f64<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f64)> implements @sub  {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f64
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f64
  %2 = llvm.fsub %0, %1 : f64
  %3 = meta.cast_from_builtin %2 : f64 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// mul
//===----------------------------------------------------------------------===//

kgen.generator.interface @mul<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @mul_f32<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @mul {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
  %2 = llvm.fmul %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

kgen.generator @mul_f64<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f64)> implements @mul {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f64
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f64
  %2 = llvm.fmul %0, %1 : f64
  %3 = meta.cast_from_builtin %2 : f64 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// div
//===----------------------------------------------------------------------===//

kgen.generator.interface @div<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @div_f32<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @div {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
  %2 = llvm.fdiv %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

kgen.generator @div_f64<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f64)> implements @div {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f64
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f64
  %2 = llvm.fdiv %0, %1 : f64
  %3 = meta.cast_from_builtin %2 : f64 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// float_constant
//===----------------------------------------------------------------------===//

kgen.generator.interface @float_constant<value: f64, type: dtype>() -> !meta.scalar<type>

kgen.generator @float_constant_f64<value: f64, type: dtype>() -> !meta.scalar<type>
  constraints <eq_dtype(type, f64)> implements @float_constant  {
  %0 = kgen.param.value : f64 = <value>
  %1 = meta.cast_from_builtin %0: f64 to !meta.scalar<type>
  kgen.return %1 : !meta.scalar<type>
}

kgen.generator @float_constant_f32<value: f64, type: dtype>() -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @float_constant  {
  %0 = kgen.param.value : f64 = <value>
  %1 = llvm.fptrunc %0 : f64 to f32
  %2 = meta.cast_from_builtin %1: f32 to !meta.scalar<type>
  kgen.return %2 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// buffer.load
//===----------------------------------------------------------------------===//

kgen.generator.interface @buffer.load<size, type: dtype>(%buffer: !meta.buffer<size, type>, %idx: index) -> !meta.scalar<type>

kgen.generator @buffer_load_f32<size, type: dtype>(%buffer: !meta.buffer<size, type>, %idx: index) -> !meta.scalar<type>
  implements @buffer.load {
  %ptr = meta.buffer.address %buffer: !meta.buffer<size, type>
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

kgen.generator.interface @buffer.store<size, type: dtype>(%value: !meta.scalar<type>, %buffer: !meta.buffer<size, type>, %idx: index) -> ()

// TODO: Currently, the signature of @buffer_store_f32 must exactly match the signature of @buffer_store, so we can't
// have the signature use f32. When we allow this, we shoudl change the signature to be specialized to f32
kgen.generator @buffer_store_f32<size, type: dtype>(%value: !meta.scalar<type>, %buffer: !meta.buffer<size, type>, %idx: index) -> ()
  implements @buffer.store {
  %ptr = meta.buffer.address %buffer: !meta.buffer<size, type>
  %llvm_ptr = builtin.unrealized_conversion_cast %ptr: !meta.pointer<type> to !llvm.ptr<f32>
  %i64_idx = builtin.unrealized_conversion_cast %idx: index to i64
  %scalar_val = meta.cast_to_builtin %value: !meta.scalar<type> to f32
  %element_ptr = llvm.getelementptr %llvm_ptr[%i64_idx]: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  llvm.store %scalar_val, %element_ptr : !llvm.ptr<f32>
  kgen.return
}

//===----------------------------------------------------------------------===//
// fma
//===----------------------------------------------------------------------===//

// Perform an FMA operation. The fma operation performs an `a*b + c` operation.
kgen.generator.interface @fma<type: dtype>(%a: !meta.scalar<type>, %b: !meta.scalar<type>, %c: !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @fma_f32<type: dtype>(%a: !meta.scalar<type>, %b: !meta.scalar<type>, %c: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @fma {
  %0 = meta.cast_to_builtin %a: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %b: !meta.scalar<type> to f32
  %2 = meta.cast_to_builtin %c: !meta.scalar<type> to f32
  %3 = "llvm.intr.fma"(%0, %1, %2) : (f32, f32, f32) -> f32
  %4 = meta.cast_from_builtin %3 : f32 to !meta.scalar<type>
  kgen.return %4 : !meta.scalar<type>
}

kgen.generator @fma_f64<type: dtype>(%a: !meta.scalar<type>, %b: !meta.scalar<type>, %c: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f64)> implements @fma {
  %0 = meta.cast_to_builtin %a: !meta.scalar<type> to f64
  %1 = meta.cast_to_builtin %b: !meta.scalar<type> to f64
  %2 = meta.cast_to_builtin %c: !meta.scalar<type> to f64
  %3 = "llvm.intr.fma"(%0, %1, %2) : (f64, f64, f64) -> f64
  %4 = meta.cast_from_builtin %3 : f64 to !meta.scalar<type>
  kgen.return %4 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// horner evaluator
//===----------------------------------------------------------------------===//

// The horner(val, coeffs) where val is a scalar and coeffs is a list of
// coefficients  [c0, c1, c2, ..., cn] is defined by the following equation:
// horner(val, coeffs) = c0 + val * (c1 + val * (c2 + val * (... + val * cn)))
//                     = fma(val, horner(val, coeffs[1:]), c0)
kgen.generator.interface @horner<size, type: dtype>(%val: !meta.scalar<type>, %coefficients: !meta.buffer<size, type>) -> !meta.scalar<type>

kgen.generator @horner_f32<size, type: dtype>(%val: !meta.scalar<type>, %coefficients: !meta.buffer<size, type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @horner {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %zerofVal = kgen.call @float_constant<value : f64 = 0.0, type : dtype = type>() : () -> !meta.scalar<type>
  %zerof = meta.cast_to_builtin %zerofVal: !meta.scalar<type> to f32
  %numCoeffs = meta.buffer.size %coefficients: !meta.buffer<size, type>
  %res = scf.for %i = %zero to %numCoeffs step %one iter_args(%sum = %zerof) -> (f32) {
    %sumVal = meta.cast_from_builtin %sum : f32 to !meta.scalar<type>
    %coeff = kgen.call @buffer.load<size = size, type : dtype = type>(%coefficients, %i): (!meta.buffer<size, type>, index) -> !meta.scalar<type>
    %resVal = kgen.call @fma<type : dtype = type>(%sumVal, %val, %coeff) : (!meta.scalar<type>, !meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
    %res = meta.cast_to_builtin %resVal: !meta.scalar<type> to f32
    scf.yield %res : f32
  }
  %result = meta.cast_from_builtin %res : f32 to !meta.scalar<type>
  kgen.return %result : !meta.scalar<type>
}
