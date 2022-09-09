// RUN: kgen %s -execute -kernel="run_exp_kernel:f32():%t_run_exp.o" -I %S/../kernels | FileCheck %s -check-prefix=EXEC
// RUN: kgen %s -emit -kernel="exp_f32_kernel:f32(f32):%t_expf32.o" -I %S/../kernels
// COM: Check the object file.
// RUN: llvm-objdump %t_expf32.o -t | FileCheck %s -check-prefix=OBJ
// COM: Check the header file.
// RUN: cat %t_expf32.h | FileCheck %s -check-prefix=HDR

kgen.include "library.mlir"

kgen.generator.interface @exp<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>

kgen.generator @exp_f32(%arg0: f32) -> f32 {
  %0 = meta.cast_from_builtin %arg0 : f32 to !meta.scalar<f32>
  %1 = kgen.call @exp<type: dtype = f32>(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f32> to f32
  kgen.return %2 : f32
}

kgen.generator @exp_f64(%arg0: f64) -> f64 {
  %0 = meta.cast_from_builtin %arg0 : f64 to !meta.scalar<f64>
  %1 = kgen.call @exp<type: dtype = f64>(%0) : (!meta.scalar<f64>) -> !meta.scalar<f64>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f64> to f64
  kgen.return %2 : f64
}

// COM: run_exp_kernel computes exp(1.0)
kgen.generator @run_exp() -> f32 {
  %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
  %1 = kgen.call @exp_f32(%0) : (f32) -> f32
  kgen.return %1 : f32
}

// EXEC: --- 'run_exp_kernel' returned 2.7{{[0-9]+}}

// OBJ-LABEL: SYMBOL TABLE
// OBJ-DAG: F {{.*}}exp_f32_kernel
// OBJ-DAG: F {{.*}}exp_intrinsic_f32,type=f32
// OBJ-DAG: *UND* {{.*}}expf

// HDR-LABEL: extern float exp_f32_kernel(float);
