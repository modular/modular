// RUN: not kgen-execute %s -execute -kernel="unhandled:i31():%t_unhandled.o" 2>&1 >/dev/null | FileCheck -check-prefix=BADSIG %s

// BADSIG: unhandled signature: i31()
llvm.func @unhandled() -> i31 {
  %0 = llvm.mlir.constant(1 : i31) : i31
  llvm.return %0 : i31
}
