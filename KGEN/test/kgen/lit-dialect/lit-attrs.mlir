// RUN: kgen-opt %s -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #lit.none : !lit.none
"a"() {a = #lit.none : !lit.none} : () -> ()

lit.struct.decl @Foo {
  lit.struct.field foo : index
  lit.struct.field bar : !kgen.dtype
}

// CHECK: #lit.struct<{foo = 5, bar: dtype = f32}>
"a"() {a = #lit.struct<{foo = 5, bar: dtype = f32}> : !kgen.declref<@Foo>} : () -> ()

// CHECK: #lit.lifetime : !lit.lifetime
"a"() {a = #lit.lifetime : !lit.lifetime} : () -> ()


kgen.generator @lifetime_lower<p: !lit.lifetime>(%a: !lit.lifetime) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @caller
kgen.generator @caller() {
  // CHECK: %lifetime = kgen.param.constant: lifetime = <#lit.lifetime>
  %cst = kgen.param.constant: lifetime = <#lit.lifetime>
  // CHECK: kgen.call @lifetime_lower<:lifetime #lit.lifetime>(%lifetime) : (!lit.lifetime) -> ()
  kgen.call @lifetime_lower<:lifetime #lit.lifetime>(%cst) : (!lit.lifetime) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type<p: lifetime>(
// CHECK-SAME: %arg0: !lit.ref<@Foo, p>
// CHECK-SAME: %arg1: !lit.ref<mut @Foo, p>)
kgen.generator @ref_type<p: !lit.lifetime>(%a: !lit.ref<@Foo, p>,
                                           %b: !lit.ref<mut @Foo, p>) {
  kgen.return
}
