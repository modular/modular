// RUN: kgen-opt %s -lower-lit -split-input-file -verify-diagnostics -mlir-print-debuginfo | FileCheck %s

// CHECK: (ctor_fn)foo
// CHECK-NEXT: kgen.return
// CHECK: (dtor_fn)foo
// CHECK-NEXT: kgen.return
// CHECK: kgen.global @foo : index (0, @"(ctor_fn)foo", @"(dtor_fn)foo")
lit.globalvar.decl @foo : index {
}, {
}

// CHECK: (ctor_fn)bar
lit.globalvar.decl @bar : index {
  // CHECK-NEXT: pop.global.address @foo
  lit.globalvar.ref @foo : <index>
  // CHECK-NEXT: pop.global.address @baz
  lit.globalvar.ref @baz : <index>
  // CHECK-NEXT: kgen.return
}, {
}
// CHECK: kgen.global @bar : index (2,

// CHECK: kgen.global @baz : index (1,
lit.globalvar.decl @baz : index {
  lit.globalvar.ref @foo : <index>
}, {
}

// CHECK: kgen.global @boo : index (3,
lit.globalvar.decl @boo : index {
  lit.globalvar.ref @bar : <index>
  lit.globalvar.ref @baz : <index>
}, {
}

// -----

lit.file_module @module {
  // CHECK: kgen.global export @foo : index
  lit.globalvar.decl export @exported : index attributes {linkageName = "foo"} {}, {}

  // CHECK-LABEL: kgen.generator @"module::ref_exported"
  lit.func @ref_exported() {
    // CHECK-NEXT: pop.global.address @foo : <index>
    %0 = lit.globalvar.ref @module::@exported : <index>
    kgen.return
  }
}

// -----
// expected-error @-2 {{cyclic dependencies between global variables in 'lower-lit' pass}}

lit.globalvar.decl @foo : index {
  lit.globalvar.ref @bar : <index>
}, {
}

lit.globalvar.decl @bar : index {
  lit.globalvar.ref @foo : <index>
}, {
}

// -----

// CHECK: kgen.generator @"(ctor_fn)self"
lit.globalvar.decl @self : index {
  // CHECK-NEXT: pop.global.address @self
  lit.globalvar.ref @self : <index>
}, {
  lit.globalvar.ref @self : <index>
}

// -----

#file = #debuginfo.file<"foo.mlir" in "/">

// CHECK-LABEL: kgen.generator @"(ctor_fn)foo"()
// CHECK-NEXT:    pop.global.address @foo : <index> loc(#[[LOC_CTOR_OP:.*]])
// CHECK-NEXT:    kgen.return loc(#[[LOC_CTOR:.*]])
// CHECK-NEXT:  } loc(#[[LOC_CTOR]])
// CHECK-LABEL: kgen.generator @"(dtor_fn)foo"() {
// CHECK-NEXT:    pop.global.address @foo : <index> loc(#[[LOC_DTOR_OP:.*]])
// CHECK-NEXT:    kgen.return loc(#[[LOC_DTOR:.*]])
// CHECK-NEXT:  } loc(#[[LOC_DTOR]])
// CHECK-NEXT:  kgen.global @foo : index (0, @"(ctor_fn)foo", @"(dtor_fn)foo") loc(#[[LOC_OP:.*]])
lit.globalvar.decl @foo : index {
  lit.globalvar.ref @foo : <index> loc(fused<#file>["foo.mlir":9:4])
}, {
  lit.globalvar.ref @foo : <index> loc(fused<#file>["foo.mlir":10:4])
} loc(fused<#file>["foo.mlir":8:4])

// CHECK-DAG: #[[FILE:.*]] = #debuginfo.file<"foo.mlir" in "/">
// CHECK-DAG: #[[COMPILE_UNIT:.*]] = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #[[FILE]], producer = "kgen", isOptimized = true, emissionKind = Full>
// CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<() -> (): DW_CC_normal>
// CHECK-DAG: #[[SP_CTOR:.*]] = #debuginfo.subprogram<compileUnit = #[[COMPILE_UNIT]], scope = #[[FILE]], name = "(ctor_fn)foo", linkageName = "(ctor_fn)foo", file = #[[FILE]], line = 8, scopeLine = 8, subprogramFlags = Definition> : ![[SP_TYPE]]
// CHECK-DAG: #[[SP_DTOR:.*]] = #debuginfo.subprogram<compileUnit = #[[COMPILE_UNIT]], scope = #[[FILE]], name = "(dtor_fn)foo", linkageName = "(dtor_fn)foo", file = #[[FILE]], line = 8, scopeLine = 8, subprogramFlags = Definition> : ![[SP_TYPE]]

// CHECK-DAG: #[[LOC1:.*]] = loc("foo.mlir":8:4)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mlir":9:4)
// CHECK-DAG: #[[LOC3:.*]] = loc("foo.mlir":10:4)

// CHECK-DAG: #[[LOC_CTOR]] = loc(fused<#[[SP_CTOR]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CTOR_OP]] = loc(fused<#[[SP_CTOR]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_DTOR]] = loc(fused<#[[SP_DTOR:.*]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_DTOR_OP]] = loc(fused<#[[SP_DTOR:.*]]>[#[[LOC3]]])
