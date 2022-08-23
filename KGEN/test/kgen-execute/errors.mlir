// RUN: not kgen-execute %s -execute -kernel="exec_exp:f32():%t_exec_exp.o" -verify-diagnostics 2>&1 >/dev/null | FileCheck %s

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

kgen.kernel @exec_exp() -> f32 {
  %0 = llvm.call @"float_constant_indirect"() : () -> f32
  %1 = llvm.call @"exp_intrinsic_f32,type=f32"(%0) : (f32) -> f32
  kgen.return %1 : f32
}

// CHECK: could not find local callee '@float_constant_f32,value=1,type=f32' in the current module.
