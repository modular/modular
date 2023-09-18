// RUN: kgen-opt -lower-lit -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT:  }
lit.func @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @varDecl
// CHECK-SAME:  (%[[ARG0:.*]]: index) -> index {
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    kgen.return %[[ARG0]] : index
// CHECK-NEXT:  }

lit.func @varDecl(%arg0: index) -> index {
  %a = lit.varlet.decl "a" var : <index>
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.generator @varDecl2
// CHECK-SAME:  (%[[ARG0:.*]]: index) {
// CHECK-NEXT: kgen.param.declare alife: lifetime = <#lit.lifetime>
// CHECK-NEXT: %0 = pop.stack_allocation 1 x index
// CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0 : !kgen.pointer<index> to !lit.ref<mut index, alife>
// CHECK-NEXT: kgen.return
lit.func @varDecl2(%arg0: index) {
  %a = lit.varlet.decl2 "a" var : !lit.ref<mut index, alife>
  kgen.return
}

// CHECK-LABEL: kgen.generator @letDecl(%arg0
// CHECK-NEXT:    kgen.return %arg0 : index
lit.func @letDecl(%arg0: index) -> index {
  %a = lit.letreg.decl "a" = %arg0 : index
  %b = lit.letreg.decl "b" = %a : index
  kgen.return %b : index
}

lit.func @decorator() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @decorated_fn
lit.func @decorated_fn()
  // CHECK-NEXT: decorators <:() -> () @decorator>
  decorators<:() -> () @decorator> {
  kgen.return
}

//===----------------------------------------------------------------------===//
// Aliases
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: kgen.generator @aliasDecls()
// CHECK-NEXT: kgen.param.declare aliasDecl = <5>
// CHECK-NEXT: kgen.return
lit.func @aliasDecls() {
  lit.alias.decl aliasDecl = <5>
  lit.alias.fwd.decl "aliasFwdDecl" : index
  kgen.return
}

//===----------------------------------------------------------------------===//
// Structs
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @Adder<size> {
  // CHECK-LABEL: kgen.generator @"Adder::__add__"<size>(%arg0: !kgen.declref<@Adder<size = size>>) {
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.func @__add__(%self: !kgen.declref<@Adder<size = size>>)  {
    %0 = lit.varlet.decl "a" var : <index>
    %one = index.constant 1
    pop.store %one, %0 : !kgen.pointer<index>
    kgen.return
  }
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"

// CHECK-LABEL: lit.struct.decl @A
lit.struct.decl @A {
  lit.func @foo(%self: !kgen.declref<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @"B::foo"
// CHECK-NEXT: call_param[(!kgen.declref<@A>) -> (): @"A::foo"]

// CHECK-LABEL: lit.struct.decl @B
lit.struct.decl @B {
  lit.func @foo(%self: !kgen.declref<@B>, %a: !kgen.declref<@A>) {
    kgen.call_param[(!kgen.declref<@A>) -> (): @A::@foo](%a)
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !kgen.declref<@A>, %b: !kgen.declref<@B>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@B>, !kgen.declref<@A>) -> (): @"B::foo"]
  kgen.call_param[(!kgen.declref<@B>, !kgen.declref<@A>) -> (): @B::@foo](%b, %a)
  // CHECK-NEXT: constant: (!kgen.declref<@A>) -> () = <@"A::foo">
  %0 = kgen.param.constant: (!kgen.declref<@A>) -> () = <@A::@foo>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"<N, M>

// CHECK-LABEL: lit.struct.decl @A<N>
lit.struct.decl @A<N> {
  lit.func @foo<M>(%self: !kgen.declref<@A<N = N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !kgen.declref<@A<N = 1>>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@A<N = 1>>) -> index: @"A::foo"<1, 2>]
  %0 = kgen.call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<1, 2>](%a)
  kgen.return
}

// -----

lit.struct.decl @A {
}

// CHECK: kgen.generator @rhslitdeclref_no_params(%arg0: !kgen.declref<@A>) {
lit.func @rhslitdeclref_no_params(%x: !kgen.declref<@A>) {
  kgen.return
}

// -----

lit.struct.decl @A<b, c> {
}

// CHECK: kgen.generator @rhslitdeclref_params(%arg0: !kgen.declref<@A<b = 10, c = 11>>) {
lit.func @rhslitdeclref_params(%x: !kgen.declref<@A<b = 10, c = 11>>) {
  kgen.return
}

// -----

lit.struct.decl @A {
  lit.func @B() {
    kgen.return
  }
}

// CHECK-LABEL: @callIt
lit.func @callIt() {
  // CHECK-NEXT: kgen.call @"A::B"
  kgen.call @A::@B() : () -> ()
  kgen.return
}

// -----

// CHECK-NOT: lit.alias.decl
lit.alias.decl A = <1>
lit.struct.decl @foo {
  // CHECK-NOT: lit.alias.decl
  lit.alias.decl B = <2>
 // CHECK-LABEL:  @"foo::f"() -> index {
  lit.func @f() -> index {
    // CHECK: kgen.param.declare
    lit.alias.decl C = <3>
    %0 = kgen.param.constant: index = <1>
    kgen.return %0 : index
  }
}

//===----------------------------------------------------------------------===//
// HandleVariant
//===----------------------------------------------------------------------===//

lit.func @throwing_caller() throws -> !pop.variant<@Error, !lit.none> {
  %y = lit.varlet.decl "y" : <@MyStruct>
  %0 = kgen.call @throwing_callee(%y) : (!kgen.pointer<@MyStruct> byref_result) throws -> !pop.variant<@Error, !lit.none>
  // CHECK: %2 = pop.variant.is !pop.array<0, i1>, %1 : !pop.variant<@Error, array<0, i1>>
  // CHECK: %3 = hlcf.if %2 -> !pop.array<0, i1> {
  // CHECK:   %4 = pop.variant.get %1 : !pop.variant<@Error, array<0, i1>> as !pop.array<0, i1>
  // CHECK:   hlcf.yield %4 : !pop.array<0, i1>
  // CHECK: } else {
  // CHECK:   %4 = pop.variant.get %1 : !pop.variant<@Error, array<0, i1>> as !kgen.declref<@Error>
  // CHECK:   %5 = pop.variant.create %4 : !kgen.declref<@Error> -> !pop.variant<@Error, array<0, i1>>
  // CHECK:   kgen.return %5 : !pop.variant<@Error, array<0, i1>>
  // CHECK:  }
  %1 = lit.handle_variant %0, %y : (!pop.variant<@Error, !lit.none>, !kgen.pointer<@MyStruct>) -> !lit.none {
    %7 = pop.variant.get %0 : !pop.variant<@Error, !lit.none> as !lit.none
    lit.yield %7 : !lit.none
  } else {
    %8 = pop.variant.get %0 : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
    %9 = pop.variant.create %8 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    kgen.return %9 : !pop.variant<@Error, !lit.none>
  }
  %6 = kgen.param.constant: !pop.variant<@Error, !lit.none> = <#pop.variant<:!lit.none #lit.none>>
  kgen.return %6 : !pop.variant<@Error, !lit.none>
}

lit.func @caller_reg() -> !lit.none {
  lit.try {
    %0 = kgen.call @throwing_callee() : () throws -> !pop.variant<@Error, index>
    // CHECK: %[[VAR0:.*]] = pop.variant.is index, %0 : !pop.variant<@Error, index>
    // CHECK: %[[VAR1:.*]] = hlcf.if %[[VAR0]] -> index {
    // CHECK:   %[[VAR2:.*]] = pop.variant.get %0 : !pop.variant<@Error, index> as index
    // CHECK:   hlcf.yield %[[VAR2]] : index
    // CHECK: } else {
    // CHECK:   %[[VAR3:.*]] = pop.variant.get %0 : !pop.variant<@Error, index> as !kgen.declref<@Error>
    // CHECK:   lit.raise %[[VAR3]] : <@Error>
    // CHECK:   kgen.unreachable
    // CHECK: }
    %1 = lit.handle_variant %0 : (!pop.variant<@Error, index>) -> index {
      %7 = pop.variant.get %0 : !pop.variant<@Error, index> as index
      lit.yield %7 : index
    } else {
      %8 = pop.variant.get %0 : !pop.variant<@Error, index> as !kgen.declref<@Error>
      lit.raise %8 : !kgen.declref<@Error>
      kgen.unreachable
    }
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  }
  %6 = kgen.param.constant: !lit.none = <#lit.none>
  kgen.return %6 : !lit.none
}

//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//

lit.struct.decl @Error {}

lit.func @throwing_func() throws -> !pop.variant<@Error, !lit.none> {
  %1 = lit.struct.create() : () -> !kgen.declref<@Error>
  %2 = pop.variant.create %1 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
  // CHECK: kgen.return %1 : !pop.variant<@Error, array<0, i1>>
  lit.error_return %2 : !pop.variant<@Error, !lit.none>
}

// CHECK-LABEL: kgen.generator @return_raise_or
// CHECK-SAME: -> !pop.variant<@Error, array<0, i1>>
lit.func @return_raise_or(%cond: i1, %err: !kgen.declref<@Error>) -> !pop.variant<@Error, !lit.none> {
  hlcf.if %cond {
    // CHECK: %[[ERR:.*]] = pop.variant.create %arg1
    %0 = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    // CHECK-NEXT: kgen.return %[[ERR]]
    kgen.return %0 : !pop.variant<@Error, !lit.none>
  } else {
    hlcf.yield
  }

  %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK: %[[VAL:.*]] = pop.variant.create %{{.*}}
  %1 = pop.variant.create %0 : !lit.none -> !pop.variant<@Error, !lit.none>
  // CHECK-NEXT: kgen.return %[[VAL]]
  kgen.return %1 : !pop.variant<@Error, !lit.none>
}

// CHECK-LABEL: kgen.generator @removeMetadata
// CHECK-SAME: (%arg0: !kgen.pointer<index>) throws ->
lit.func @removeMetadata(%arg0: !kgen.pointer<index> byref) throws -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// -----

// CHECK: (ctor_fn)foo
// CHECK-NEXT: kgen.return
// CHECK: (dtor_fn)foo
// CHECK-NEXT: kgen.return
// CHECK: kgen.global @foo : index [@"(ctor_fn)foo", @"(dtor_fn)foo"](0)
lit.globalvar.decl @foo : index {
}, {
}

// CHECK: (ctor_fn)bar
lit.globalvar.decl @bar : index {
  // CHECK-NEXT: kgen.global.address @foo
  lit.globalvar.ref @foo : <index>
  // CHECK-NEXT: kgen.global.address @baz
  lit.globalvar.ref @baz : <index>
  // CHECK-NEXT: kgen.return
}, {
}
// CHECK: kgen.global @bar : index [{{.*}}](2)

// CHECK: kgen.global @baz : index [{{.*}}](1)
lit.globalvar.decl @baz : index {
  lit.globalvar.ref @foo : <index>
}, {
}

// CHECK: kgen.global @boo : index [{{.*}}](3)
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
    // CHECK-NEXT: kgen.global.address @foo : <index>
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
  // CHECK-NEXT: kgen.global.address @self
  lit.globalvar.ref @self : <index>
}, {
  lit.globalvar.ref @self : <index>
}

//===----------------------------------------------------------------------===//
// Modules
//===----------------------------------------------------------------------===//

// -----

// CHECK-NOT: lit.file_module

lit.file_module @module {
  // CHECK: kgen.generator @"module::test"()
  lit.func @test()  {
    kgen.return
  }

  lit.struct.decl @Adder<size> {
    // CHECK-LABEL: kgen.generator @"module::Adder::__add__"<size>(%arg0: !kgen.declref<@"module::Adder"<size = size>>) {
    // CHECK-NEXT:    kgen.call @"module::test"() : () -> ()
    lit.func @__add__(%self: !kgen.declref<@module::@Adder<size = size>>)  {
      kgen.call @module::@test() : () -> ()
      kgen.return
    }
  }

  // CHECK-LABEL: lit.struct.decl @"module::Adder"<size> {
}

// CHECK-LABEL: kgen.generator @caller(%arg0: !kgen.declref<@"module::Adder"<size = 10>>)
lit.func @caller(%ref: !kgen.declref<@module::@Adder<size = 10>>)  {
  // CHECK: kgen.call @"module::Adder::__add__"
  kgen.call @module::@Adder::@__add__<10>(%ref) : (!kgen.declref<@module::@Adder<size = 10>>) -> ()
  kgen.return
}

// -----

// CHECK-NOT: lit.
lit.package @package {
  lit.file_module @module {
    // CHECK: kgen.link "lib.a" as @"package::module::lib"
    kgen.link "lib.a" as @lib

    // CHECK: kgen.generator export @"package::module::foo"()
    lit.func export @foo() {
      kgen.return
    }
  }
}

// -----

lit.file_module @module {
  // CHECK-NOT: lit.alias.decl
  lit.alias.decl A = <42>
}
