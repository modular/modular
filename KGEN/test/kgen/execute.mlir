// RUN: kgen %s -execute -func="exec_exp:f32()" -func="void:()" | FileCheck %s

kgen.func @"exp_intrinsic_f32,type=f32"(%arg0: f32) -> f32 {
  %0 = "llvm.intr.exp"(%arg0) : (f32) -> f32
  kgen.return %0 : f32
}
kgen.func @exp_f32(%arg0: f32) -> f32 {
  %0 = kgen.call @"exp_intrinsic_f32,type=f32"(%arg0) : (f32) -> f32
  kgen.return %0 : f32
}
kgen.func @"float_constant_f32,value=1,type=f32"() -> f32 {
  %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %1 = llvm.fptrunc %0 : f64 to f32
  kgen.return %1 : f32
}

kgen.func @"float_constant_indirect"() -> f32 {
  %0 = kgen.call @"float_constant_f32,value=1,type=f32"() : () -> f32
  kgen.return %0 : f32
}

kgen.func @exec_exp() -> f32 {
  %0 = kgen.call @"float_constant_indirect"() : () -> f32
  %1 = kgen.call @"exp_intrinsic_f32,type=f32"(%0) : (f32) -> f32
  kgen.return %1 : f32
}

kgen.func @void() {
  %0 = kgen.call @"float_constant_indirect"() : () -> f32
  kgen.return
}

kgen.export @exec_exp
kgen.export @void

// COM: exec_exp computes exp(1.0)
// CHECK: --- 'exec_exp' returned 2.7{{[0-9]+}}
// CHECK: --- 'void' finished
