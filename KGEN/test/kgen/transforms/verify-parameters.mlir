// RUN: kgen-opt %s -split-input-file -verify-parameters=simplify=true -kgen-print-inline-type-values | FileCheck %s

// CHECK-LABEL: no_constrains_deduplication
kgen.generator @no_constrains_deduplication() {
  kgen.param.declare cond = <1>
  kgen.param.if <eq(cond, 1)> {
    kgen.param.declare B0 : !kgen.string = <"foo">
    // CHECK: kgen.param.assert <0>, "foo"
    kgen.param.assert <eq(2, 3)>, B0
    kgen.return
  } else {
    kgen.param.declare B1 : !kgen.string = <"bar">
    // CHECK: kgen.param.assert <0>, "bar"
    kgen.param.assert <eq(2, 3)>, B1
    kgen.param.yield
  }
  kgen.param.declare B2 : !kgen.string = <"baz">
  // CHECK: kgen.param.assert <0>, "baz"
  kgen.param.assert <eq(2, 3)>, B2
  kgen.return
}

// -----

// Test get_witness contextual evaluation under LIT, which is required for
// checking the symbol use of `expect_associated_alias` in the
// `use_associated_alias` function is correct.

!Fooable = !lit.trait<@Fooable>

lit.trait.decl @Fooable<?, SELF: !Fooable> {
  lit.alias.decl MyType: !kgen.type
}

#wrapper_index = #kgen.type<!lit.struct<@Wrapper<:type index>>> : !Fooable

lit.struct.decl @"Wrapper"<T: type> {
  lit.struct.field data: !kgen.param<T>
  lit.alias.decl MyType: !kgen.type = <index>
  kgen.conformance @Fooable {
    kgen.witness "MyType" : !kgen.type = #kgen.type<index>
  }
}

// CHECK-LABEL: lit.fn @expect_associated_alias
// CHECK-SAME:    (%arg: !kgen.param<#kgen.get_witness<#kgen.param.decl.ref<"T"> : !lit.trait<@Fooable>, "Fooable", "MyType" : !kgen.string>>)
lit.fn @expect_associated_alias<T: !Fooable>(%arg: !kgen.param<#kgen.get_witness<#kgen.param.decl.ref<"T"> : !Fooable, "Fooable", "MyType" : !kgen.string>>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @use_associated_alias
lit.fn @use_associated_alias(%arg: index) -> !kgen.none {
  // CHECK-NEXT: lit.call @expect_associated_alias<:trait<@Fooable> @Wrapper<:type index>>(%arg) : !lit.generator<("arg": index) -> !kgen.none>
  %none = lit.call @expect_associated_alias<:!Fooable #wrapper_index>(%arg) : !lit.generator<("arg": index) -> !kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// -----

// Test get_witness contextual evaluation under KGEN too.

kgen.struct.generator @Wrapper<T: type> = struct_inst<"Wrapper"[T]<:type T>(data: typevalue<T>)> {
  kgen.conformance @Fooable {
    kgen.witness "MyType" : !kgen.type = #kgen.type<index>
  }
}

#wrapper_index = #kgen.type<typevalue<#kgen.genref<@Wrapper<:type index>>>, struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.generator @expect_associated_alias
// CHECK-SAME:    (%arg0: !kgen.param<#kgen.get_witness<#kgen.param.decl.ref<"T"> : !kgen.type, "Fooable", "MyType" : !kgen.string>>)
kgen.generator @expect_associated_alias<T: !kgen.type>(%arg: !kgen.param<#kgen.get_witness<#kgen.param.decl.ref<"T"> : !kgen.type, "Fooable", "MyType" : !kgen.string>>) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @use_associated_alias
kgen.generator @use_associated_alias(%arg: index) -> !kgen.none {
  // CHECK-NEXT: kgen.call @expect_associated_alias<:type [typevalue<#kgen.genref<@Wrapper<:type index>>>, struct<(index)>]>(%arg0)
  %none = kgen.call @expect_associated_alias<:!kgen.type #wrapper_index>(%arg) : !kgen.generator<(index) -> !kgen.none>
  kgen.return %none : !kgen.none
}
