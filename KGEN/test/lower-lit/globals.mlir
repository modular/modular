// RUN: kgen-opt %s -lower-lit -split-input-file -verify-diagnostics | FileCheck %s

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
