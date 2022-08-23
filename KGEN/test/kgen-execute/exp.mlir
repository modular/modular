// RUN: kgen-execute %s -execute -kernel="exec_exp:f32():%t_exec_exp.o" | FileCheck %s

llvm.func @"exp_intrinsic_f32,type=f32"(%arg0: f32) -> f32 {
  %0 = "llvm.intr.exp"(%arg0) : (f32) -> f32
  llvm.return %0 : f32
}
llvm.func @exp_f32(%arg0: f32) -> f32 {
  %0 = llvm.call @"exp_intrinsic_f32,type=f32"(%arg0) : (f32) -> f32
  llvm.return %0 : f32
}
llvm.func @"float_constant_f32,value=1,type=f32"() -> f32 {
  %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %1 = llvm.fptrunc %0 : f64 to f32
  llvm.return %1 : f32
}

llvm.func @"float_constant_indirect"() -> f32 {
  %0 = llvm.call @"float_constant_f32,value=1,type=f32"() : () -> f32
  llvm.return %0 : f32
}

kgen.kernel @exec_exp() -> f32 {
  %0 = llvm.call @"float_constant_indirect"() : () -> f32
  %1 = llvm.call @"exp_intrinsic_f32,type=f32"(%0) : (f32) -> f32
  kgen.return %1 : f32
}

// COM: exec_exp computes exp(1.0)
// CHECK: --- Kernel 'exec_exp' returned 2.7{{[0-9]+}}

