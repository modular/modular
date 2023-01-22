// RUN: kgen-opt %s -allow-unregistered-dialect | FileCheck %s

// CHECK: #lit.none : i32
"a"() {a = #lit.none : i32} : () -> ()

lit.struct.decl @Foo {
  lit.struct.field foo : index
  lit.struct.field bar : !kgen.dtype
}

// CHECK: #lit.struct<{foo = 5, bar: dtype = f32}>
"a"() {a = #lit.struct<{foo = 5, bar: dtype = f32}> : !kgen.declref<@Foo>} : () -> ()
