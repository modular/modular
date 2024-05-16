// RUN: kgen %s -execute -func="my_exported_kernel:f32(f32)" | FileCheck %s -check-prefix=EXEC
// RUN: kgen %s -emit -o %t_my_kernel.o
// COM: Check the object file.
// RUN: llvm-objdump %t_my_kernel.o -t | FileCheck %s -check-prefix=OBJ
// COM: Check the header file.
// RUN: kgen %s -emit-header | FileCheck %s -check-prefix=HDR

kgen.func export C @my_exported_kernel(%arg0: f32) -> f32 {
  kgen.return %arg0 : f32
}

kgen.func @noop() {
  kgen.return
}

kgen.global export @exported_global : i32 [@noop, @noop](0)

// EXEC: --- 'my_exported_kernel' returned 1.0

// OBJ-LABEL: {{.*}}my_kernel.o
// OBJ-LABEL: {{.*}}my_kernel.o
// OBJ-LABEL: {{.*}}my_kernel.o
// OBJ-LABEL: {{.*}}my_kernel.o
// OBJ-LABEL: {{.*}}my_kernel.o
// OBJ: my_exported_kernel
// OBJ-LABEL: ({{.*}}kgen-exec-obj.mlir.5.o):
// OBJ: exported_global

// HDR-LABEL: extern float my_exported_kernel(float);
