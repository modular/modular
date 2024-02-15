// RUN: kgen-opt -check-recursive-structs %s -verify-diagnostics

// expected-error @+1 {{struct contains recursive reference to itself}}
lit.struct.decl @Bar {
  // expected-error @+1 {{recursive nested struct field, try adding indirection to recursive reference}}
  lit.struct.field x : !lit.declref<@Foo>
}

// expected-error @+1 {{struct contains recursive reference to itself}}
lit.struct.decl @Foo {
  lit.struct.field x : !lit.declref<@Bar>
}
