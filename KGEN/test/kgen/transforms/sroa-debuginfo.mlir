
// RUN: kgen-opt -sroa -split-input-file %s | FileCheck %s

!subroutine = !debuginfo.subroutine<() -> (): DW_CC_normal>
!unresolved = !debuginfo.unresolved<!kgen.pointer<struct<index, index>>>
#file = #debuginfo.file<"/tmp/test.mojo" in "/">
#subprogram = #debuginfo.subprogram<name = "__next__"> : !subroutine
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "self", file = #file, line = 27, arg = 1> : !unresolved

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

// CHECK: !unresolved = !debuginfo.unresolved<!kgen.pointer<index>>
// CHECK: #[[VAR0:.*]] = #debuginfo.local_variable<{{.*}}, name = "self.0", {{.*}}> : !unresolved
// CHECK: #[[VAR1:.*]] = #debuginfo.local_variable<{{.*}}, name = "self.1", {{.*}}> : !unresolved

// CHECK-LABEL: @sroa_valueop
kgen.func @sroa_valueop() {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: %1 = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x !pop.struct<index, index> loc(#loc)
  // CHECK-NEXT: debuginfo.value #[[VAR0]] = %0 : !kgen.pointer<index>
  // CHECK-NEXT: debuginfo.value #[[VAR1]] = %1 : !kgen.pointer<index>
  debuginfo.value #local_variable = %0 : !kgen.pointer<struct<index, index>> loc(#loc)
  // CHECK-NEXT: kgen.return
  kgen.return loc(#loc)
} loc(#loc)

// -----

#file = #debuginfo.file<"foo.mojo" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "max", linkageName = "max", file = #file, line = 0, scopeLine = 0, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #sp, name = "x", file = #file, line = 0, arg = 1> : !debuginfo.unresolved<!pop.struct<index, index>>

#loc = loc(fused<#sp>["foo.mojo":0:0])

// CHECK: [[VAR0:#.*]] = #debuginfo.local_variable<{{.*}}, name = "x.0", {{.*}}> : !unresolved
// CHECK: [[VAR1:#.*]] = #debuginfo.local_variable<{{.*}}, name = "x.1", {{.*}}> : !unresolved

// CHECK-LABEL: @load_debug_var
kgen.func @load_debug_var(%arg0: !pop.struct<index, index>) {
  // CHECK-COUNT-2: pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x struct<index, index> loc(#loc)
  pop.store %arg0, %0 : !kgen.pointer<struct<index, index>> loc(#loc)
  %1 = pop.load %0 : !kgen.pointer<struct<index, index>> loc(#loc)
  // CHECK: [[VALUE0:%.*]] = pop.load
  // CHECK-NEXT: debuginfo.value [[VAR0]] = [[VALUE0]]
  // CHECK: [[VALUE1:%.*]] = pop.load
  // CHECK-NEXT: debuginfo.value [[VAR1]] = [[VALUE1]]
  debuginfo.value #local_variable = %1 : !pop.struct<index, index> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)
