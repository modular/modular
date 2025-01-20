// RUN: kgen %s -emit -o %t_my_kernel.o --save-temps --temps-dir=%t_temps
// COM: Check the save tmp files.
// RUN: find $(dirname %t_temps) -type f -name "*.s" -print -quit | xargs cat | FileCheck %s -check-prefix=ASM
// RUN: find $(dirname %t_temps) -type f -name "*.pre-split.*.ll" -print -quit | xargs cat | FileCheck %s -check-prefix=PRESPLIT

kgen.func export C @my_exported_kernel(%arg0: f32) -> f32 {
  kgen.return %arg0 : f32
}

kgen.func @noop() {
  kgen.return
}

kgen.global export @exported_global : i32 [@noop, @noop](0)

// ASM-DAG: .section
// ASM-DAG: KGEN_EE_JIT_GlobalDestructor

// PRESPLIT-LABEL: ; ModuleID = 'kgen-save-temps.mlir'
// PRESPLIT-DAG: @exported_global = global i32 undef
// PRESPLIT-DAG: define weak void @KGEN_EE_JIT_GlobalConstructor() #0
// PRESPLIT-DAG: define weak void @KGEN_EE_JIT_GlobalDestructor() #0
// PRESPLIT-DAG: define dso_local float @my_exported_kernel(float noundef %0) #0
// PRESPLIT-DAG: define internal void @noop() #0
