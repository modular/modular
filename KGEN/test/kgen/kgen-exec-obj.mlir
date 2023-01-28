// RUN: kgen %s -execute -func="exp_f32:f32(f32)" -I %S/../kernels | FileCheck %s -check-prefix=EXEC
// RUN: kgen %s -emit -o %t_expf32.o -I %S/../kernels
// COM: Check the object file.
// RUN: llvm-objdump %t_expf32.o -t | FileCheck %s -check-prefix=OBJ
// COM: Check the header file.
// RUN: cat %t_expf32.h | FileCheck %s -check-prefix=HDR

kgen.include "library.mlir"

kgen.generator.interface @exp<type: dtype>(%x: !pop.simd<1, type>) -> !pop.simd<1, type>

kgen.generator @exp_f32(%arg0: f32) -> f32 {
  %0 = pop.cast_from_builtin %arg0 : f32 to !pop.simd<1, f32>
  %1 = kgen.call @exp<type: dtype = f32>(%0) : (!pop.simd<1, f32>) -> !pop.simd<1, f32>
  %2 = pop.cast_to_builtin %1 : !pop.simd<1, f32> to f32
  kgen.return %2 : f32
}

kgen.generator @exp_f64(%arg0: f64) -> f64 {
  %0 = pop.cast_from_builtin %arg0 : f64 to !pop.simd<1, f64>
  %1 = kgen.call @exp<type: dtype = f64>(%0) : (!pop.simd<1, f64>) -> !pop.simd<1, f64>
  %2 = pop.cast_to_builtin %1 : !pop.simd<1, f64> to f64
  kgen.return %2 : f64
}

kgen.export @exp_f32 as @my_exp_f32

// COM: We have exp_f32 compute exp(1.0) for this test.
// EXEC: --- 'exp_f32' returned 2.7{{[0-9]+}}

// OBJ-LABEL: SYMBOL TABLE
// OBJ-DAG: F {{.*}}exp_f32
// OBJ-DAG: F {{.*}}my_exp_f32_c

// HDR-LABEL: extern float my_exp_f32_c(float);
