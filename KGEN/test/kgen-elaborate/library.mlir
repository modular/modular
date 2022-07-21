// This is the library of kernel generators used by the tests in this directory.
// It is also run as a test, which just verifies that it parses correctly.

// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -o /dev/null

// CHECK-LABEL: kgen.generator.interface @unary_add<size>(si32) -> si32

// expected-note @+1 {{library interface}}
kgen.generator.interface @unary_add<size>(si32) -> si32

kgen.generator @unary_add_library_impl1<size>(%arg0: si32) -> si32
  implements @unary_add {

  // Silly op so we know when something used this.
  "unary_add_library_impl1"() : () -> ()

  // TODO: Do something with <size>
  kgen.return %arg0 : si32
}


kgen.generator.interface @scalar_add<dt: dtype>(!meta.scalar<dt>, !meta.scalar<dt>) -> !meta.scalar<dt>

kgen.generator @scalar_add_i32<dt: dtype>(%arg0: !meta.scalar<dt>, %arg1: !meta.scalar<dt>) -> !meta.scalar<dt>
  implements @scalar_add {
    %0 = meta.cast_to_builtin %arg0: !meta.scalar<dt> to i32
    %1 = meta.cast_to_builtin %arg1: !meta.scalar<dt> to i32
    %2 = arith.addi %0, %1 : i32
    %3 = meta.cast_from_builtin %2 : i32 to !meta.scalar<dt>
    kgen.return %3 : !meta.scalar<dt>
  }


kgen.generator @scalar_add_i64<dt: dtype>(%arg0: !meta.scalar<dt>, %arg1: !meta.scalar<dt>) -> !meta.scalar<dt>
  implements @scalar_add {
    %0 = meta.cast_to_builtin %arg0: !meta.scalar<dt> to i64
    %1 = meta.cast_to_builtin %arg1: !meta.scalar<dt> to i64
    %2 = arith.addi %0, %1 : i64
    %3 = meta.cast_from_builtin %2 : i64 to !meta.scalar<dt>
    kgen.return %3 : !meta.scalar<dt>
  }


kgen.generator @scalar_add_f32<dt: dtype>(%arg0: !meta.scalar<dt>, %arg1: !meta.scalar<dt>) -> !meta.scalar<dt>
  implements @scalar_add {
    %0 = meta.cast_to_builtin %arg0: !meta.scalar<dt> to f32
    %1 = meta.cast_to_builtin %arg1: !meta.scalar<dt> to f32
    %2 = arith.addf %0, %1 : f32
    %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<dt>
    kgen.return %3 : !meta.scalar<dt>
  }


kgen.generator @scalar_add_f64<dt: dtype>(%arg0: !meta.scalar<dt>, %arg1: !meta.scalar<dt>) -> !meta.scalar<dt>
  implements @scalar_add {
    %0 = meta.cast_to_builtin %arg0: !meta.scalar<dt> to f64
    %1 = meta.cast_to_builtin %arg1: !meta.scalar<dt> to f64
    %2 = arith.addf %0, %1 : f64
    %3 = meta.cast_from_builtin %2 : f64 to !meta.scalar<dt>
    kgen.return %3 : !meta.scalar<dt>
  }

// Arithmetics operations.
// TODO: Add support for types other than f32.
kgen.generator.interface @sub<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @sub_impl<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @sub  {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
  %2 = llvm.fsub %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

kgen.generator.interface @mul<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @mul_impl<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @mul {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
  %2 = llvm.fmul %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

kgen.generator.interface @div<type: dtype>(!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @div_impl<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> implements @div {
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<type> to f32
  %1 = meta.cast_to_builtin %arg1: !meta.scalar<type> to f32
  %2 = llvm.fdiv %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<type>
  kgen.return %3 : !meta.scalar<type>
}

// Compute erf as Taylor series expansion: erf(x) = 2/sqrt(pi) * (x - x^3/3)

kgen.generator @scalar_erf<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>
  constraints <eq_dtype(type, f32)> {
  // Compute 2/sqrt(pi) * (x - x^3 / 3) as 2/sqrt(pi) * x * (1 - x^2 / 3)
  %sqrt_of_pi_f32 = arith.constant 1.77245384 : f32
  %sqrt_of_pi = meta.cast_from_builtin %sqrt_of_pi_f32 : f32 to !meta.scalar<type>
  %one_f32 = arith.constant 1.0 : f32
  %one = meta.cast_from_builtin %one_f32 : f32 to !meta.scalar<type>
  %two_f32 = arith.constant 2.0 : f32
  %two = meta.cast_from_builtin %two_f32 : f32 to !meta.scalar<type>
  %three_f32 = arith.constant 3.0 : f32
  %three = meta.cast_from_builtin %three_f32 : f32 to !meta.scalar<type>
  %fact1   = kgen.call @div<type : dtype = f32>(%two, %sqrt_of_pi) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %x_sqr   = kgen.call @mul<type : dtype = f32>(%x, %x) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %x_sqr_3 = kgen.call @div<type : dtype = f32>(%x_sqr, %three) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %fact3   = kgen.call @sub<type : dtype = f32>(%one, %x_sqr_3) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %prod1   = kgen.call @mul<type : dtype = f32>(%fact1, %x) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  %prod2   = kgen.call @mul<type : dtype = f32>(%prod1, %fact3) : (!meta.scalar<type>, !meta.scalar<type>) -> !meta.scalar<type>
  kgen.return %prod2 : !meta.scalar<type>
}
