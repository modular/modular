// RUN: kgen %s -execute -func="my_exported_kernel:f32(f32)" | FileCheck %s -check-prefix=EXEC
// RUN: kgen %s -emit -o %t_my_kernel.o
// COM: Check the object file.
// RUN: llvm-objdump %t_my_kernel.o -t | FileCheck %s -check-prefix=OBJ
// COM: Check the header file.
// RUN: kgen %s -emit-header | FileCheck %s -check-prefix=HDR

kgen.generator @my_kernel(%arg0: f32) -> f32 {
  kgen.return %arg0 : f32
}

kgen.export @my_kernel to C as @my_exported_kernel

// EXEC: --- 'my_exported_kernel' returned 1.0

// OBJ-LABEL: SYMBOL TABLE
// OBJ-DAG: my_exported_kernel

// HDR-LABEL: extern float my_exported_kernel(float);
