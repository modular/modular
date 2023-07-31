// RUN: kgen-opt -lower-lit -mlir-print-debuginfo %s | FileCheck %s

// CHECK: ![[DIVAR_TYPE:.*]] = !debuginfo.unresolved<!pop.pointer<index>>
// CHECK: ![[DILETVAR_TYPE:.*]] = !debuginfo.unresolved<index>
// CHECK: #[[DISP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = "varDecl", linkageName = "Int::varDecl", file = #{{.*}}, line = 1, scopeLine = 1, subprogramFlags = Definition>
// CHECK: #[[DIVAR:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "a", file = #{{.*}}, line = 10, arg = 0> : ![[DIVAR_TYPE]]
// CHECK: #[[DILETVAR:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "let_value", file = #{{.*}}, line = 11, arg = 0> : ![[DILETVAR_TYPE]]

// CHECK-LABEL: kgen.generator @"Int::varDecl"
// CHECK-SAME: (%[[ARG0:.*]]: index
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    debuginfo.value #[[DIVAR]] = %[[VAR_A]] : !pop.pointer<index>
// CHECK-NEXT:    debuginfo.value #[[DILETVAR]] = %[[ARG0]] : index

// CHECK-LABEL: kgen.generator @"module::fn"()

// CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = "fn", linkageName = "module::fn", file = #{{.*}}, line = 1, scopeLine = 1, subprogramFlags = Definition>

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "LIT", isOptimized = true, emissionKind = Full>
#sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "varDecl", linkageName = "varDecl", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#module_sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "fn", linkageName = "fn", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#loc = loc("test.mlir":10:6)

lit.struct.decl @Int {
  lit.func @varDecl(%arg0: index) -> index {
    %a = lit.varlet.decl "a", var = true, synth=false : <index> loc(fused<#sp>["test.mlir":10:10])
    %let_value = lit.letreg.decl "let_value" = %arg0 : index loc(fused<#sp>["test.mlir":11:10])
    kgen.return %let_value : index loc(fused<#sp>[#loc])
  } loc(fused<#sp>[#loc])
}

lit.file_module @module {
  lit.func @fn() {
    kgen.return loc(fused<#module_sp>[#loc])
  } loc(fused<#module_sp>[#loc])
}
