// RUN: kgen-opt -lower-lit -mlir-print-debuginfo %s | FileCheck %s

#file = #debuginfo.file<"foo.mlir" in "/">

// CHECK-LABEL: kgen.generator @"(ctor_fn)foo"()
// CHECK-NEXT:    %0 = kgen.global.address @foo : <index> loc(#[[LOC_CTOR_OP:.*]])
// CHECK-NEXT:    kgen.return loc(#[[LOC_CTOR:.*]])
// CHECK-NEXT:  } loc(#[[LOC_CTOR]])
// CHECK-LABEL: kgen.generator @"(dtor_fn)foo"()
// CHECK-NEXT:    %0 = kgen.global.address @foo : <index> loc(#[[LOC_DTOR_OP:.*]])
// CHECK-NEXT:    kgen.return loc(#[[LOC_DTOR:.*]])
// CHECK-NEXT:  } loc(#[[LOC_DTOR]])
// CHECK-NEXT:  kgen.global @foo : index [@"(ctor_fn)foo", @"(dtor_fn)foo"](0) loc(#[[LOC_OP:.*]])
lit.globalvar.decl @foo : index {
  lit.globalvar.ref @foo : <index, mut #lit.any.origin> loc(fused<#file>["foo.mlir":9:4])
}, {
  lit.globalvar.ref @foo : <index, mut #lit.any.origin> loc(fused<#file>["foo.mlir":10:4])
} loc(fused<#file>["foo.mlir":8:4])

// CHECK-DAG: #[[FILE:.*]] = #debuginfo.file<"foo.mlir" in "/">
// CHECK-DAG: #[[COMPILE_UNIT:.*]] = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #[[FILE]], producer = "kgen", isOptimized = true, emissionKind = Full
// CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<() -> (): DW_CC_normal>
// CHECK-DAG: #[[SP_CTOR:.*]] = #debuginfo.subprogram<compileUnit = #[[COMPILE_UNIT]], scope = #[[FILE]], sourceName = <"(ctor_fn)foo">, linkageName = "(ctor_fn)foo", file = #[[FILE]], line = 8, scopeLine = 8, subprogramFlags = Definition> : ![[SP_TYPE]]
// CHECK-DAG: #[[SP_DTOR:.*]] = #debuginfo.subprogram<compileUnit = #[[COMPILE_UNIT]], scope = #[[FILE]], sourceName = <"(dtor_fn)foo">, linkageName = "(dtor_fn)foo", file = #[[FILE]], line = 8, scopeLine = 8, subprogramFlags = Definition> : ![[SP_TYPE]]

// CHECK-DAG: #[[LOC1:.*]] = loc("foo.mlir":8:4)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mlir":9:4)
// CHECK-DAG: #[[LOC3:.*]] = loc("foo.mlir":10:4)

// CHECK-DAG: #[[LOC_CTOR]] = loc(fused<#[[SP_CTOR]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CTOR_OP]] = loc(fused<#[[SP_CTOR]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_DTOR]] = loc(fused<#[[SP_DTOR:.*]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_DTOR_OP]] = loc(fused<#[[SP_DTOR:.*]]>[#[[LOC3]]])
