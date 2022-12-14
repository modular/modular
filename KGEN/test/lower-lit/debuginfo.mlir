// RUN: kgen-opt -lower-lit -mlir-print-debuginfo %s | FileCheck %s

// CHECK: ![[DIVAR_TYPE:.*]] = !debuginfo.unresolved<!pop.pointer<index>>
// CHECK: #[[DISP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = "varDecl", linkageName = "Int::varDecl", file = #{{.*}}, line = 1, scopeLine = 1, subprogramFlags = Definition> : !subroutine
// CHECK: #[[DIVAR:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "a", file = #{{.*}}, line = 10, arg = 0> : ![[DIVAR_TYPE]]

// CHECK-LABEL: kgen.generator @"Int::varDecl"
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    debuginfo.value #[[DIVAR]] = %[[VAR_A]] : !pop.pointer<index>

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "LIT", isOptimized = true, emissionKind = Full>
#sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "varDecl", linkageName = "varDecl", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>

kgen.struct.decl @Int {
  lit.func @varDecl(%arg0: index) -> index {
    %a = lit.var.decl "a" : <index> loc(fused<#sp>["test.mlir":10:10])
    kgen.return %arg0 : index
  } loc(fused<#sp>[unknown])
}
