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
