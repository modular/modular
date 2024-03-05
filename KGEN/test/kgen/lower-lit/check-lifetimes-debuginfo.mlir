// RUN: kgen-opt -check-lifetimes -split-input-file -mlir-print-debuginfo %s | FileCheck %s

// CHECK: ![[DI_PTR_TYPE:.*]] = !debuginfo.ti.ptr<index>
// CHECK: ![[DI_INDEX_TYPE:.*]] = !debuginfo.unresolved<index>
// CHECK: #[[DIEXPR_IRVALUE:.*]] = #debuginfo.expr.irvalue : ![[DI_PTR_TYPE]]
// CHECK: #[[DIEXPR_DEREF:.*]] = #debuginfo.expr.deref<#[[DIEXPR_IRVALUE]]> : ![[DI_INDEX_TYPE]]
// CHECK: #[[DISP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = <"varDecl">, linkageName = "varDecl", file = #{{.*}}, line = 1, scopeLine = 1, subprogramFlags = Definition>
// CHECK: #[[DIVAR:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "a", file = #{{.*}}, line = 10> : ![[DI_INDEX_TYPE]]

// CHECK-NOT: #debuginfo.local_variable<scope = #[[DISP]], name = "b", file = #{{.*}}, line = 12> : ![[DI_INDEX_TYPE]]

// CHECK-LABEL: lit.func @varDecl
// CHECK-SAME: (%[[ARG0:.*]]: index loc
// CHECK-NEXT:    %[[VAR_A:.*]] = lit.var.decl "a"
// CHECK-NEXT:    debuginfo.value #[[DIVAR]] #[[DIEXPR_DEREF]] = %[[VAR_A]] : !lit.ref<index, mut life_a>
// CHECK-NEXT:    %[[VAR_B:.*]] = lit.var.decl "b"
// CHECK-NEXT:    kgen.return

// CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = <"fn">, linkageName = "fn", file = #{{.*}}, line = 1, scopeLine = 1, subprogramFlags = Definition>

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "LIT", isOptimized = true, emissionKind = Full>
#sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = <"varDecl">, linkageName = "varDecl", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#module_sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = <"fn">, linkageName = "fn", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#loc = loc("test.mlir":10:6)

lit.struct.decl @Int {
  lit.func @varDecl(%arg0: index) -> index {
    %a = lit.var.decl "a" var : !lit.ref<index, mut life_a> loc(fused<#sp>["test.mlir":10:10])
    %b = lit.var.decl "b" synth : !lit.ref<index, mut life_b> loc(fused<#sp>["test.mlir":12:10])
    kgen.return %arg0 : index loc(fused<#sp>[#loc])
  } loc(fused<#sp>[#loc])
}

lit.file_module @module {
  lit.func @fn() {
    kgen.return loc(fused<#module_sp>[#loc])
  } loc(fused<#module_sp>[#loc])
}
