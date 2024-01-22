// RUN: kgen-opt -lower-lit -split-input-file -verify-diagnostics %s | FileCheck %s

// FIXME(https://github.com/modularml/modular/issues/28335): This file contains
// many busted examples that don't pass -verify-parameters.  :-(

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @calls(%arg0
lit.func @calls[imm a, mut b](%arg0: !lit.signature<[2]() -> ()>) {
  // CHECK: kgen.call @calls() : () -> ()
  lit.call @calls[imm a, mut b]() : !lit.signature<[2]() -> ()>
  // CHECK: kgen.call_param[() -> (): @calls]()
  lit.call_param[!lit.signature<[2]() -> ()>: @calls][imm a, mut b]()
  // CHECK: kgen.call_signature %arg0() : () -> ()
  lit.call_signature %arg0[imm a, mut b]() : !lit.signature<[2]() -> ()>
  kgen.return
}

// CHECK-LABEL: kgen.generator @async_call()
lit.func @async_call[imm a, mut b]() async {
  // CHECK: lit.async.call[() async -> (): @async_call]()
  lit.async.call[!lit.signature<[2]() async -> ()>: @async_call][imm a, mut b]()
  kgen.return
}

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32 owned) -> si32
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }
lit.func @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @varDecl
// CHECK-SAME:  (%[[ARG0:.*]]: index owned) -> index
// CHECK-NEXT:    kgen.param.declare life: lifetime
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0
// CHECK-NEXT:    kgen.return %[[ARG0]] : index
// CHECK-NEXT:  }

lit.func @varDecl(%arg0: index) -> index {
  %a = lit.varlet.decl "a" var : !lit.ref<index, mut *"life">
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.generator @varDecl2
// CHECK-SAME:  (%[[ARG0:.*]]: index owned)
// CHECK-NEXT: kgen.param.declare alife: lifetime<1> = <#lit.lifetime>
// CHECK-NEXT: %0 = pop.stack_allocation 1 x index
// CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0 : !kgen.pointer<index> to !lit.ref<index, mut alife>
// CHECK-NEXT: kgen.return
lit.func @varDecl2(%arg0: index) {
  %a = lit.varlet.decl "a" var : !lit.ref<index, mut alife>
  kgen.return
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

// CHECK-LABEL: @generic_types_retain_convention
lit.func @generic_types_retain_convention<T: type>(
  // CHECK: %arg0: !kgen.paramref<T> borrow,
  // CHECK: %arg1: !kgen.pointer<T> byref,
  // CHECK: %arg2: !kgen.paramref<T> owned,
  // CHECK: %arg3: index borrow,
  // CHECK: %arg4: !kgen.pointer<index> owned
  %p: !kgen.paramref<T> borrow,
  %q: !kgen.pointer<T> byref,
  %r: !kgen.paramref<T> owned,
  %s1: index borrow,
  %s2: !kgen.pointer<index> owned
){
  kgen.return
}

lit.func @generic_callee<T: type>(%p: !kgen.paramref<T> borrow){
  kgen.return
}

// CHECK-LABEL: @call_generic
lit.func @call_generic(%p: index borrow) {
  // CHECK: kgen.call @generic_callee<:type index>({{.*}}) : (index borrow) -> ()
  kgen.call @generic_callee<:type index>(%p) : !lit.signature<(index borrow) -> ()>
  kgen.return
}


//===----------------------------------------------------------------------===//
// Nested Functions
//===----------------------------------------------------------------------===//

lit.struct.decl @StructWithNestedFn<a_param> {
  // CHECK-LABEL: kgen.generator @"StructWithNestedFn::topLevelFunction"<a_param, b_param>() -> index
  lit.func @topLevelFunction<b_param>() -> index {
    // CHECK: kgen.param.declare.region nestedFunction = () -> index
    lit.func nestedFunction() -> index {
      kgen.unreachable
    }
    // CHECK: kgen.param.declare b: () -> index = <nestedFunction>
    kgen.param.declare b: !lit.signature<() -> index> = <nestedFunction>

    // CHECK: kgen.param.declare.region paramNestedFunc = <c_param>()
    lit.func paramNestedFunc<c_param>() {
      kgen.return
    }
    // CHECK: kgen.param.declare c: <[] -> index>() -> () = <bind_signature(:<index -> index>() -> () paramNestedFunc, 2)>
    kgen.param.declare c: !lit.signature<<[] -> index>() -> ()> = <bind_signature(:!lit.signature<<index -> index>() -> ()> paramNestedFunc, 2)>

    %idx0_0 = index.constant 0
    kgen.return %idx0_0 : index
  }
}

// CHECK-LABEL: lit.struct.decl @StructWithNestedFn<a_param>

// CHECK-LABEL: kgen.generator @topFunc
lit.func @topFunc() {
  // CHECK: kgen.param.declare.region midFunc
  lit.func midFunc() {
    // CHECK: kgen.param.declare.region botFunc
    lit.func botFunc() {
      kgen.return
    }
    // CHECK: declare bot: () -> () = <botFunc>
    kgen.param.declare bot: !lit.signature<() -> ()> = <botFunc>
    kgen.return
  }
  // CHECK: declare mid: () -> () = <midFunc>
  kgen.param.declare mid: !lit.signature<() -> ()> = <midFunc>
  kgen.return
}

//===----------------------------------------------------------------------===//
// Structs
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @Adder<size> {
  // CHECK-LABEL: kgen.generator @"Adder::__add__"<size>(%arg0: !kgen.declref<@Adder<size>> owned)
  // CHECK-NEXT:    kgen.param.declare life: lifetime
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.func @__add__(%self: !kgen.declref<@Adder<size>>)  {
    %0 = lit.varlet.decl "a" var : !lit.ref<index, mut *"life">
    %one = index.constant 1
    lit.ref.store %one, %0 : !lit.ref<index, mut *"life">
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
  lit.func @foo<M>(%self: !kgen.declref<@A<N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !kgen.declref<@A<1>>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@A<1>>) -> index: @"A::foo"<1, 2>]
  %0 = kgen.call_param[(!kgen.declref<@A<1>>) -> index: @A::@foo<1, 2>](%a)
  kgen.return
}

// -----

lit.struct.decl @A {
}

// CHECK: kgen.generator @rhslitdeclref_no_params(%arg0: !kgen.declref<@A> owned)
lit.func @rhslitdeclref_no_params(%x: !kgen.declref<@A>) {
  kgen.return
}

// -----

lit.struct.decl @A<b, c> {
}

// CHECK: kgen.generator @rhslitdeclref_params(%arg0: !kgen.declref<@A<10, 11>> owned)
lit.func @rhslitdeclref_params(%x: !kgen.declref<@A<10, 11>>) {
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
 // CHECK-LABEL:  @"foo::f"() -> index
  lit.func @f() -> index {
    // CHECK: kgen.param.declare
    lit.alias.decl C = <3>
    %0 = kgen.param.constant: index = <1>
    kgen.return %0 : index
  }
}

// -----

//===----------------------------------------------------------------------===//
// HandleVariant
//===----------------------------------------------------------------------===//

lit.func @throwing_caller() throws -> !kgen.variant<@Error, none> {
  %y = lit.varlet.decl "y" let : !lit.ref<@MyStruct, mut *"life">
  %yp = lit.ref.to_pointer %y : !lit.ref<@MyStruct, mut *"life">
  %0 = kgen.call @throwing_callee(%yp) : (!kgen.pointer<@MyStruct> byref_result) throws -> !kgen.variant<@Error, none>
  // CHECK: [[V:%.*]] = kgen.call @throwing_callee(
  // CHECK: [[VAR0:%.*]] = kgen.variant.is [[V]], 1 : <@Error, none>
  // CHECK:  = hlcf.if [[VAR0]] -> !kgen.none {
  // CHECK:   [[VAR1:%.*]] = kgen.variant.take [[V]], 1 : <@Error, none>
  // CHECK:   hlcf.yield [[VAR1]] : !kgen.none
  // CHECK: } else {
  // CHECK:   [[VAR2:%.*]] = kgen.variant.take [[V]], 0 : <@Error, none>
  // CHECK:   [[VAR3:%.*]] = kgen.variant.create [[VAR2]], 0 : <@Error, none>
  // CHECK:   kgen.return [[VAR3]]
  // CHECK:  }
  %1 = lit.handle_variant %0, %yp : (!kgen.variant<@Error, none>, !kgen.pointer<@MyStruct>) -> !kgen.none {
    %7 = kgen.variant.take %0, 1 : <@Error, none>
    lit.yield %7 : !kgen.none
  } else {
    %8 = kgen.variant.take %0, 0 : <@Error, none>
    %9 = kgen.variant.create %8, 0 : <@Error, none>
    kgen.return %9 : !kgen.variant<@Error, none>
  }
  %6 = kgen.param.constant: !kgen.variant<@Error, none> = <#kgen.variant<:!kgen.none #kgen.none, 1>>
  kgen.return %6 : !kgen.variant<@Error, none>
}

lit.func @caller_reg() -> !kgen.none {
  lit.try {
    %0 = kgen.call @throwing_callee() : () throws -> !kgen.variant<@Error, index>
    // CHECK: [[VAR0:%.*]] = kgen.variant.is %0, 1 : <@Error, index>
    // CHECK: [[VAR1:%.*]] = hlcf.if [[VAR0]] -> index {
    // CHECK:   [[VAR2:%.*]] = kgen.variant.take %0, 1 : <@Error, index>
    // CHECK:   hlcf.yield [[VAR2]] : index
    // CHECK: } else {
    // CHECK:   [[VAR3:%.*]] = kgen.variant.take %0, 0 : <@Error, index>
    // CHECK:   lit.raise [[VAR3]] : <@Error>
    // CHECK:   kgen.unreachable
    // CHECK: }
    %1 = lit.handle_variant %0 : (!kgen.variant<@Error, index>) -> index {
      %7 = kgen.variant.take %0, 1 : <@Error, index>
      lit.yield %7 : index
    } else {
      %8 = kgen.variant.take %0, 0 : <@Error, index>
      lit.raise %8 : !kgen.declref<@Error>
      kgen.unreachable
    }
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  }
  %6 = kgen.param.constant: none = <#kgen.none>
  kgen.return %6 : !kgen.none
}

// -----

//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//
lit.struct.decl @Error {}

lit.func @throwing_func() throws -> !kgen.variant<@Error, none> {
  %1 = lit.struct.create() : () -> !kgen.declref<@Error>
  %2 = kgen.variant.create %1, 0 : <@Error, none>
  // CHECK: kgen.return %1 : !kgen.variant<@Error, none>
  lit.error_return %2 : !kgen.variant<@Error, none>
}

// CHECK-LABEL: kgen.generator @return_raise_or
// CHECK-SAME: -> !kgen.variant<@Error, none>
lit.func @return_raise_or(%cond: i1, %err: !kgen.declref<@Error>) -> !kgen.variant<@Error, none> {
  hlcf.if %cond {
    // CHECK: %[[ERR:.*]] = kgen.variant.create %arg1
    %0 = kgen.variant.create %err, 0 : <@Error, none>
    // CHECK-NEXT: kgen.return %[[ERR]]
    kgen.return %0 : !kgen.variant<@Error, none>
  } else {
    hlcf.yield
  }

  %0 = kgen.param.constant: none = <#kgen.none>
  // CHECK: %[[VAL:.*]] = kgen.variant.create %{{.*}}
  %1 = kgen.variant.create %0, 1 : <@Error, none>
  // CHECK-NEXT: kgen.return %[[VAL]]
  kgen.return %1 : !kgen.variant<@Error, none>
}

// CHECK-LABEL: kgen.generator @removeMetadata
// CHECK-SAME: (%arg0: !kgen.pointer<index> byref) throws ->
lit.func @removeMetadata(%arg0: !kgen.pointer<index> byref) throws -> !kgen.variant<@Error, index> {
  %0 = index.constant 0
  %1 = kgen.variant.create %0, 1 : <@Error, index>
  kgen.return %1 : !kgen.variant<@Error, index>
}

// -----

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// CHECK: kgen.generator{{.*}}(ctor_fn)foo
// CHECK-NEXT: kgen.return
// CHECK: kgen.generator{{.*}}(dtor_fn)foo
// CHECK-NEXT: kgen.return
// CHECK: kgen.global @foo : index [@"(ctor_fn)foo", @"(dtor_fn)foo"](0)
lit.globalvar.decl @foo : index {
}, {
}

// CHECK: (ctor_fn)bar
lit.globalvar.decl @bar : index {
  // CHECK-NEXT: %0 = kgen.global.address @foo
  // CHECK-NEXT:  builtin.unrealized_conversion_cast %0
  lit.globalvar.ref @foo : <index, mut #lit.lifetime>
  // CHECK-NEXT: %2 = kgen.global.address @baz
  // CHECK-NEXT:  builtin.unrealized_conversion_cast %2
  lit.globalvar.ref @baz : <index, mut #lit.lifetime>
  // CHECK-NEXT: kgen.return
}, {
}
// CHECK: kgen.global @bar : index [{{.*}}](2)

// CHECK: kgen.global @baz : index [{{.*}}](1)
lit.globalvar.decl @baz : index {
  lit.globalvar.ref @foo : <index, mut #lit.lifetime>
}, {
}

// CHECK: kgen.global @boo : index [{{.*}}](3)
lit.globalvar.decl @boo : index {
  lit.globalvar.ref @bar : <index, mut #lit.lifetime>
  lit.globalvar.ref @baz : <index, mut #lit.lifetime>
}, {
}

// -----

lit.file_module @module {
  // CHECK: kgen.global export @foo : index
  lit.globalvar.decl export @exported : index attributes {linkageName = "foo"} {}, {}

  // CHECK-LABEL: kgen.generator @"module::ref_exported"
  lit.func @ref_exported() {
    // CHECK-NEXT: kgen.global.address @foo : <index>
    %0 = lit.globalvar.ref @module::@exported : <index, mut #lit.lifetime>
    kgen.return
  }
}

// -----
// expected-error @-2 {{cyclic dependencies between global variables in 'lower-lit' pass}}

lit.globalvar.decl @foo : index {
  lit.globalvar.ref @bar : <index, mut #lit.lifetime>
}, {
}

lit.globalvar.decl @bar : index {
  lit.globalvar.ref @foo : <index, mut #lit.lifetime>
}, {
}

// -----

// CHECK: kgen.generator @"(ctor_fn)self"
lit.globalvar.decl @self : index {
  // CHECK-NEXT: kgen.global.address @self
  lit.globalvar.ref @self : <index, mut #lit.lifetime>
}, {
  lit.globalvar.ref @self : <index, mut #lit.lifetime>
}

// -----

//===----------------------------------------------------------------------===//
// Modules
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.file_module

lit.file_module @module {
  // CHECK: kgen.generator @"module::test"()
  lit.func @test()  {
    kgen.return
  }

  lit.struct.decl @Adder<size> {
    // CHECK-LABEL: kgen.generator @"module::Adder::__add__"<size>(%arg0: !kgen.declref<@"module::Adder"<size>> owned)
    // CHECK-NEXT:    kgen.call @"module::test"() : () -> ()
    lit.func @__add__(%self: !kgen.declref<@module::@Adder<size>>)  {
      kgen.call @module::@test() : () -> ()
      kgen.return
    }
  }

  // CHECK-LABEL: lit.struct.decl @"module::Adder"<size> {
}

// CHECK-LABEL: kgen.generator @caller(%arg0: !kgen.declref<@"module::Adder"<10>> owned)
lit.func @caller(%ref: !kgen.declref<@module::@Adder<10>>)  {
  // CHECK: kgen.call @"module::Adder::__add__"
  kgen.call @module::@Adder::@__add__<10>(%ref) : (!kgen.declref<@module::@Adder<10>>) -> ()
  kgen.return
}

// -----

// CHECK-NOT: lit.
lit.package @package {
  lit.file_module @module {
    // CHECK: kgen.link dense_resource<lib> : tensor<1xui8> as @"package::module::lib"
    kgen.link dense_resource<lib> : tensor<1xui8> as @lib

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

// CHECK: kgen.extern.generator @extern() attributes {preCompiledModuleRef = @module}
lit.func @extern() attributes {preCompiledModuleRef = @module, preElaborationName = "extern"} {
  lit.extern_func
}

// CHECK: kgen.generator @metadata
// CHECK-SAME: LLVMMetadata = {llvm.someattr = 3 : index}
lit.func @metadata() attributes {LLVMMetadata = {llvm.someattr = 3 : index}} {
  kgen.return
}

//===----------------------------------------------------------------------===//
// Implicit lifetimes.
//===----------------------------------------------------------------------===//

// -----

// Verify that the lifetimes get correctly removed and the IR is correct.

!Mem = !kgen.declref<@Mem, !lit.metatype<@Mem>>
lit.struct.decl @Mem   {
  lit.func @"__init__($test::Mem=&)"(%self: !kgen.pointer<!Mem> init_self, |) -> !kgen.none attributes {isParametric, sourceName = "__init__", specialFnKind = 2 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

// CHECK-LABEL: kgen.generator @getThing
// CHECK-SAME:(%arg0: !lit.ref<@Mem : metatype<@Mem>, mut *[0,0]> byref_result)
lit.func @getThing[mut *"abc`"](%res: !lit.ref<!Mem, mut *"abc`"> byref_result, |) -> !kgen.none attributes {isParametric, sourceName = "getThing", specialFnKind = 0 : i8} {
  // CHECK-NEXT: kgen.param.declare *"abc`": lifetime<1> = <#lit.lifetime>
  // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : !lit.ref<@Mem : metatype<@Mem>, mut *[0,0]> to !lit.ref<@Mem : metatype<@Mem>, mut *"abc`">

  // CHECK-NEXT: kgen.param.declare.region localTest = (%arg1: !lit.ref<@Mem : metatype<@Mem>, mut *[0,0]> byref_result) capturing
  lit.func localTest[mut *"__result__`0"](%__result___0[__result__]: !lit.ref<!Mem, mut *"__result__`0"> byref_result, |) capturing -> !kgen.none attributes {sourceName = "localTest", specialFnKind = 0 : i8} {
    // CHECK-NEXT: kgen.param.declare *"__result__`0": lifetime
    // CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %arg1 : !lit.ref<@Mem : metatype<@Mem>, mut *[0,0]> to !lit.ref<@Mem : metatype<@Mem>, mut *"__result__`0">
    // CHECK-NEXT: %4 = lit.ref.to_pointer %3
    %1 = lit.ref.to_pointer %__result___0 : <!Mem, mut *"__result__`0">
    %2 = lit.call @"$test"::@Mem::@"__init__($test::Mem=&)"(%1) : !lit.signature<("self": !kgen.pointer<!Mem> init_self, |) -> !kgen.none>
    %none_1 = kgen.param.constant: none = <#kgen.none>
    kgen.return %none_1 : !kgen.none
  }
  // CHECK: }
  // CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0
  // CHECK-NEXT: %2 = kgen.call_param[(!lit.ref<@Mem : metatype<@Mem>, mut *[0,0]> byref_result) capturing -> !kgen.none: localTest](%1)
  %0 = lit.call_param[!lit.signature<[1]("__result__": !lit.ref<!Mem, mut *[0,0]> byref_result, |) capturing -> !kgen.none>: localTest][mut *"abc`"](%res)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}


// CHECK-LABEL: kgen.generator @callThing
// CHECK-SAME: (%arg0: !lit.ref<@Mem : metatype<@Mem>, mut *[0,0]> byref_result)
lit.func @callThing[mut *"__result__`"](%__result__: !lit.ref<!Mem, mut *"__result__`"> byref_result, |) -> !kgen.none attributes {isParametric, sourceName = "callThing", specialFnKind = 0 : i8} {
  // CHECK-NEXT: kgen.param.declare *"__result__`": lifetime<1> = <#lit.lifetime>
  // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : !lit.ref<@Mem : metatype<@Mem>, mut *[0,0]> to !lit.ref<@Mem : metatype<@Mem>, mut *"__result__`">
  // CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0 : !lit.ref<@Mem : metatype<@Mem>, mut *"__result__`"> to !lit.ref<@Mem : metatype<@Mem>, mut *[0,0]>
  // CHECK-NEXT: kgen.call @getThing(%1)
  %0 = lit.call @getThing[mut *"__result__`"](%__result__) : !lit.signature<[1]("__result__": !lit.ref<!Mem, mut *[0,0]> byref_result, |) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
