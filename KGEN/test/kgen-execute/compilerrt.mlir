// RUN: kgen-execute %s -execute -func="run_printInt32:():%t.o" | FileCheck %s -check-prefix=EXEC-I32
// RUN: kgen-execute %s -execute -func="run_printInt64:():%t.o" | FileCheck %s -check-prefix=EXEC-I64

kgen.func public @run_printInt32() {
  %0 = llvm.mlir.constant(42 : i32) : i32
  pop.external_call @printInt32(%0) : (i32) -> ()
  kgen.return
}

// EXEC-I32: i32: 42

kgen.func public @run_printInt64() {
  %0 = llvm.mlir.constant(42 : i64) : i64
  pop.external_call @printInt64(%0) : (i64) -> ()
  kgen.return
}

// EXEC-I64: i64: 42
