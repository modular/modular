// RUN: kgen-execute %s -run-kernel="exec_exp:f32()" -verify-diagnostics 2>&1 >/dev/null | FileCheck %s

// expected-note@+1 {{callee declared here}}
llvm.func external @"float_constant_f32,value=1,type=f32"() -> f32

llvm.func @"exp_intrinsic_f32,type=f32"(%arg0: f32) -> f32 {
  %0 = "llvm.intr.exp"(%arg0) : (f32) -> f32
  llvm.return %0 : f32
}
llvm.func @exp_f32(%arg0: f32) -> f32 {
  %0 = llvm.call @"exp_intrinsic_f32,type=f32"(%arg0) : (f32) -> f32
  llvm.return %0 : f32
}

llvm.func @"float_constant_indirect"() -> f32 {
  // expected-error@+1 {{could not find local callee '@"float_constant_f32,value=1,type=f32"' in the current module}}
  %0 = llvm.call @"float_constant_f32,value=1,type=f32"() : () -> f32
  llvm.return %0 : f32
}

llvm.func @exec_exp() -> f32 {
  %0 = llvm.call @"float_constant_indirect"() : () -> f32
  %1 = llvm.call @"exp_intrinsic_f32,type=f32"(%0) : (f32) -> f32
  llvm.return %1 : f32
}

// CHECK: could not find local callee 'float_constant_f32,value=1,type=f32' in the current module.

// RUN: not kgen-execute %s -run-kernel="exec_exp:f64()" 2>&1 >/dev/null | FileCheck -check-prefix=BADSIG %s
// BADSIG: unhandled signature: f64()

// RUN: not kgen-execute %s -run-kernel="does_not_exist:f32()" 2>&1 >/dev/null | FileCheck -check-prefix=BADKERN %s
// BADKERN: could not find kernel 'does_not_exist'

