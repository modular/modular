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
// scalar_erf
//===----------------------------------------------------------------------===//

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator @scalar_erf<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> {
  // Compute 2/sqrt(pi) * (x - x^3 / 3) as 2/sqrt(pi) * x * (1 - x^2 / 3)
  %sqrt_of_pi = kgen.call @float_constant<value : f64 = 1.77245384, type : dtype = type>() : () -> !meta.scalar<type>
  %one     = kgen.call @float_constant<value : f64 = 1.0, type : dtype = type>() : () -> !meta.scalar<type>
  %two     = kgen.call @float_constant<value : f64 = 2.0, type : dtype = type>() : () -> !meta.scalar<type>
  %three   = kgen.call @float_constant<value : f64 = 3.0, type : dtype = type>() : () -> !meta.scalar<type>
  %fact1   = kgen.call @div<type : dtype = type>(%two, %sqrt_of_pi) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %x_sqr   = kgen.call @mul<type : dtype = type>(%x, %x) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %x_sqr_3 = kgen.call @div<type : dtype = type>(%x_sqr, %three) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %fact3   = kgen.call @sub<type : dtype = type>(%one, %x_sqr_3) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %prod1   = kgen.call @mul<type : dtype = type>(%fact1, %x) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %prod2   = kgen.call @mul<type : dtype = type>(%prod1, %fact3) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  kgen.return %prod2 : !meta.scalar<type>
}

//===----------------------------------------------------------------------===//
// buffer_store
//===----------------------------------------------------------------------===//

kgen.generator.interface @buffer_store<size, dt: dtype>(%value: !meta.scalar<dt>, %buffer: !meta.buffer<size, dt>, %idx: index) -> ()

// TODO: Currently, the signature of @buffer_store_f32 must exactly match the signature of @buffer_store, so we can't
// have the signature use f32. When we allow this, we shoudl change the signature to be specialized to f32
kgen.generator @buffer_store_f32<size, dt: dtype>(%value: !meta.scalar<dt>, %buffer: !meta.buffer<size, dt>, %idx: index) -> ()
  implements @buffer_store {
  %ptr = meta.buffer.address %buffer: !meta.buffer<size, dt>
  %llvm_ptr = builtin.unrealized_conversion_cast %ptr: !meta.pointer<dt> to !llvm.ptr<f32>
  %i64_idx = builtin.unrealized_conversion_cast %idx: index to i64
  %scalar_val = meta.cast_to_builtin %value: !meta.scalar<dt> to f32
  %element_ptr = llvm.getelementptr %llvm_ptr[%i64_idx]: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  llvm.store %scalar_val, %element_ptr : !llvm.ptr<f32>
  kgen.return
}
