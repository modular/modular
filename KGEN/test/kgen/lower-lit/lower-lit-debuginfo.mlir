// RUN: kgen-opt -lower-lit -split-input-file -mlir-print-debuginfo %s | FileCheck %s

// CHECK: ![[DI_PTR_TYPE:.*]] = !debuginfo.ti.ptr<index>
// CHECK: ![[DI_INDEX_TYPE:.*]] = !debuginfo.unresolved<index>
// CHECK: #[[DIEXPR_IRVALUE:.*]] = #debuginfo.expr.irvalue : ![[DI_PTR_TYPE]]
// CHECK: #[[DIEXPR_DEREF:.*]] = #debuginfo.expr.deref<#[[DIEXPR_IRVALUE]]> : ![[DI_INDEX_TYPE]]
// CHECK: #[[DISP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = <"varDecl">, linkageName = "Int::varDecl", file = #{{.*}}, line = 1, scopeLine = 1, subprogramFlags = Definition>
// CHECK: #[[DIVAR:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "a", file = #{{.*}}, line = 10> : ![[DI_INDEX_TYPE]]

// CHECK-NOT: #debuginfo.local_variable<scope = #[[DISP]], name = "b", file = #{{.*}}, line = 12> : ![[DI_PTR_TYPE]]
// CHECK: #[[DILETVAR:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "let_value", file = #{{.*}}, line = 11> : ![[DI_INDEX_TYPE]]

// CHECK-LABEL: kgen.generator @"Int::varDecl"
// CHECK-SAME: (%[[ARG0:.*]]: index loc
// CHECK-NEXT:    kgen.param.declare life_a: lifetime
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    debuginfo.value #[[DIVAR]] #[[DIEXPR_DEREF]] = %[[VAR_A]] : !kgen.pointer<index>
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[VAR_A]]
// CHECK-NEXT:    kgen.param.declare life_b: lifetime
// CHECK-NEXT:    %[[VAR_B:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    builtin.unrealized_conversion_cast %[[VAR_B]]
// CHECK-NEXT:    debuginfo.value #[[DILETVAR]] = %[[ARG0]] : index

// CHECK-LABEL: kgen.generator @"module::fn"()

// CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = <"fn">, linkageName = "module::fn", file = #{{.*}}, line = 1, scopeLine = 1, subprogramFlags = Definition>

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "LIT", isOptimized = true, emissionKind = Full>
#sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = <"varDecl">, linkageName = "varDecl", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#module_sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = <"fn">, linkageName = "fn", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#loc = loc("test.mlir":10:6)

lit.struct.decl @Int {
  lit.func @varDecl(%arg0: index) -> index {
    %a = lit.varlet.decl "a" var : !lit.ref<mut index, *"life_a"> loc(fused<#sp>["test.mlir":10:10])
    %b = lit.varlet.decl "b" var : !lit.ref<mut index, *"life_b"> {isSynthetic} loc(fused<#sp>["test.mlir":12:10])
    %let_value = lit.letreg.decl "let_value" = %arg0 : index loc(fused<#sp>["test.mlir":11:10])
    kgen.return %let_value : index loc(fused<#sp>[#loc])
  } loc(fused<#sp>[#loc])
}

lit.file_module @module {
  lit.func @fn() {
    kgen.return loc(fused<#module_sp>[#loc])
  } loc(fused<#module_sp>[#loc])
}


// -----

#file = #debuginfo.file<"foo.mlir" in "/">

// CHECK-LABEL: kgen.generator @"(ctor_fn)foo"()
// CHECK-NEXT:    %0 = kgen.global.address @foo : <index> loc(#[[LOC_CTOR_OP:.*]])
// CHECK-NEXT:    = builtin.unrealized_conversion_cast %0
// CHECK-NEXT:    kgen.return loc(#[[LOC_CTOR:.*]])
// CHECK-NEXT:  } loc(#[[LOC_CTOR]])
// CHECK-LABEL: kgen.generator @"(dtor_fn)foo"()
// CHECK-NEXT:    %0 = kgen.global.address @foo : <index> loc(#[[LOC_DTOR_OP:.*]])
// CHECK-NEXT:    = builtin.unrealized_conversion_cast %0
// CHECK-NEXT:    kgen.return loc(#[[LOC_DTOR:.*]])
// CHECK-NEXT:  } loc(#[[LOC_DTOR]])
// CHECK-NEXT:  kgen.global @foo : index [@"(ctor_fn)foo", @"(dtor_fn)foo"](0) loc(#[[LOC_OP:.*]])
lit.globalvar.decl @foo : index {
  lit.globalvar.ref @foo : <mut index, #lit.lifetime> loc(fused<#file>["foo.mlir":9:4])
}, {
  lit.globalvar.ref @foo : <mut index, #lit.lifetime> loc(fused<#file>["foo.mlir":10:4])
} loc(fused<#file>["foo.mlir":8:4])

// CHECK-DAG: #[[FILE:.*]] = #debuginfo.file<"foo.mlir" in "/">
// CHECK-DAG: #[[COMPILE_UNIT:.*]] = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #[[FILE]], producer = "kgen", isOptimized = true, emissionKind = Full>
// CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<() -> (): DW_CC_normal>
// CHECK-DAG: #[[SP_CTOR:.*]] = #debuginfo.subprogram<compileUnit = #[[COMPILE_UNIT]], scope = #[[FILE]], name = <"(ctor_fn)foo">, linkageName = "(ctor_fn)foo", file = #[[FILE]], line = 8, scopeLine = 8, subprogramFlags = Definition> : ![[SP_TYPE]]
// CHECK-DAG: #[[SP_DTOR:.*]] = #debuginfo.subprogram<compileUnit = #[[COMPILE_UNIT]], scope = #[[FILE]], name = <"(dtor_fn)foo">, linkageName = "(dtor_fn)foo", file = #[[FILE]], line = 8, scopeLine = 8, subprogramFlags = Definition> : ![[SP_TYPE]]

// CHECK-DAG: #[[LOC1:.*]] = loc("foo.mlir":8:4)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mlir":9:4)
// CHECK-DAG: #[[LOC3:.*]] = loc("foo.mlir":10:4)

// CHECK-DAG: #[[LOC_CTOR]] = loc(fused<#[[SP_CTOR]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CTOR_OP]] = loc(fused<#[[SP_CTOR]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_DTOR]] = loc(fused<#[[SP_DTOR:.*]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_DTOR_OP]] = loc(fused<#[[SP_DTOR:.*]]>[#[[LOC3]]])
