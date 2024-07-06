// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

lit.func @callee[imm a, mut b]() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @calls
lit.func @calls[imm a, mut b]<f: !lit.signature<[2]() -> ()>>(%arg0: !lit.signature<[2]() -> ()>) {
  // CHECK: kgen.call @callee() : () -> ()
  lit.call @callee[imm a, mut b]() : !lit.signature<[2]() -> ()>
  // CHECK: kgen.call_param[() -> (): f]()
  lit.call[!lit.signature<[2]() -> ()>: f][imm a, mut b]()
  // CHECK: kgen.call_indirect %arg0() : () -> ()
  lit.call_indirect %arg0[imm a, mut b]() : !lit.signature<[2]() -> ()>
  kgen.return
}

lit.func @async_fn_throws(%err: !lit.ref<index, mut #lit.lifetime> byref_error, %res: !lit.ref<index, mut #lit.lifetime> byref_result) throws|async {
  kgen.return
}

// CHECK-LABEL: kgen.generator @async_call
lit.func @async_call[imm a, mut b]() async {
  // CHECK: co.invoke[() async -> (): @async_call]()
  lit.async.call[!lit.signature<[2]() async -> ()>: @async_call][imm a, mut b]()
  // CHECK: co.invoke[(!lit.ref<index, mut #lit.lifetime> byref_error, !lit.ref<index, mut #lit.lifetime> byref_result) throws|async -> (): @async_fn_throws]()
  lit.async.call[!lit.signature<("err": !lit.ref<index, mut #lit.lifetime> byref_error, "res": !lit.ref<index, mut #lit.lifetime> byref_result) throws|async -> ()>: @async_fn_throws]()
  kgen.return
}

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }
lit.func @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @varDecl
// CHECK-SAME:  (%[[ARG0:.*]]: index) -> index
// CHECK-NEXT:    kgen.param.declare life: lifetime
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0
// CHECK-NEXT:    kgen.return %[[ARG0]] : index
// CHECK-NEXT:  }

lit.func @varDecl(%arg0: index) -> index {
  %a = lit.var.decl "a" var : !lit.ref<index, mut *"life">
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.generator @varDecl2
// CHECK-SAME:  (%[[ARG0:.*]]: index)
// CHECK-NEXT: kgen.param.declare alife: lifetime<1> = <#lit.lifetime>
// CHECK-NEXT: %0 = pop.stack_allocation 1 x index
// CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0 : !kgen.pointer<index> to !lit.ref<index, mut alife>
// CHECK-NEXT: kgen.return
lit.func @varDecl2(%arg0: index) {
  %a = lit.var.decl "a" var : !lit.ref<index, mut alife>
  kgen.return
}

lit.func @decorator() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @decorated_fn
lit.func @decorated_fn()
  // CHECK-NEXT: decorators <:() -> () @decorator>
  decorators<:!lit.signature<() -> ()> @decorator> {
  kgen.return
}

// CHECK-LABEL: @generic_types_retain_convention
lit.func @generic_types_retain_convention[imm a]<T: type>(
  // CHECK: %arg0: !kgen.paramref<T>,
  // CHECK: %arg1: !lit.ref<T, imm *[0,0]> inout,
  // CHECK: %arg2: !kgen.paramref<T> owned,
  // CHECK: %arg3: index,
  // CHECK: %arg4: !kgen.pointer<index> owned
  %p: !kgen.paramref<T>,
  %q: !lit.ref<T, imm a> inout,
  %r: !kgen.paramref<T> owned,
  %s1: index,
  %s2: !kgen.pointer<index> owned
){
  kgen.return
}

lit.func @generic_callee<T: type>(%p: !kgen.paramref<T>){
  kgen.return
}

// CHECK-LABEL: @call_generic
lit.func @call_generic(%p: index) {
  // CHECK: kgen.call @generic_callee<:type index>({{.*}}) : (index) -> ()
  kgen.call @generic_callee<:type index>(%p) : !lit.signature<("p": index) -> ()>
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
    // CHECK: kgen.param.declare c: () -> () = <bind_signature(:<index>() -> () paramNestedFunc, 2)>
    kgen.param.declare c: !lit.signature<() -> ()> = <bind_signature(:!lit.signature<<"c_param": index>() -> ()> paramNestedFunc, 2)>

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
// Imports
//===----------------------------------------------------------------------===//

// -----

// CHECK-NOT: lit.unresolved_import
lit.file_module @nested_imports {
  lit.unresolved_import @foobar as @foo

  lit.func @func() {
    lit.unresolved_import @foobar as @foo
    kgen.return
  }
}

//===----------------------------------------------------------------------===//
// Structs
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @Adder<size> {
  // CHECK-LABEL: kgen.generator @"Adder::__add__"<size>(%arg0: !lit.struct<@Adder<size>>)
  // CHECK-NEXT:    kgen.param.declare life: lifetime
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.func @__add__(%self: !lit.struct<@Adder<size>>)  {
    %0 = lit.var.decl "a" var : !lit.ref<index, mut *"life">
    %one = index.constant 1
    lit.ref.store %one, %0 : !lit.ref<index, mut *"life">
    kgen.return
  }
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"

// CHECK-LABEL: lit.struct.decl @A
lit.struct.decl @A {
  lit.func @foo(%self: !lit.struct<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @"B::foo"
// CHECK-NEXT: call_param[(!lit.struct<@A>) -> (): @"A::foo"]

// CHECK-LABEL: lit.struct.decl @B
lit.struct.decl @B {
  lit.func @foo(%self: !lit.struct<@B>, %a: !lit.struct<@A>) {
    kgen.call_param[!lit.signature<("self": !lit.struct<@A>) -> ()>: @A::@foo](%a)
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !lit.struct<@A>, %b: !lit.struct<@B>) {
  // CHECK-NEXT: call_param[(!lit.struct<@B>, !lit.struct<@A>) -> (): @"B::foo"]
  kgen.call_param[!lit.signature<("self": !lit.struct<@B>, "a": !lit.struct<@A>) -> ()>: @B::@foo](%b, %a)
  // CHECK-NEXT: constant: (!lit.struct<@A>) -> () = <@"A::foo">
  %0 = kgen.param.constant: !lit.signature<("self": !lit.struct<@A>) -> ()> = <@A::@foo>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"<N, M>

// CHECK-LABEL: lit.struct.decl @A<N>
lit.struct.decl @A<N> {
  lit.func @foo<M>(%self: !lit.struct<@A<N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !lit.struct<@A<1>>) {
  // CHECK-NEXT: call_param[(!lit.struct<@A<1>>) -> index: @"A::foo"<1, 2>]
  %0 = kgen.call_param[!lit.signature<("self": !lit.struct<@A<1>>) -> index>: @A::@foo<1, 2>](%a)
  kgen.return
}

// -----

lit.struct.decl @A {
}

// CHECK: kgen.generator @rhslitdeclref_no_params(%arg0: !lit.struct<@A>)
lit.func @rhslitdeclref_no_params(%x: !lit.struct<@A>) {
  kgen.return
}

// -----

lit.struct.decl @A<b, c> {
}

// CHECK: kgen.generator @rhslitdeclref_params(%arg0: !lit.struct<@A<10, 11>>)
lit.func @rhslitdeclref_params(%x: !lit.struct<@A<10, 11>>) {
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
  lit.call @A::@B() : !lit.signature<() -> ()>
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
// Error
//===----------------------------------------------------------------------===//
lit.struct.decl @Error {}

lit.func @throwing_func() throws -> !kgen.variant<@Error, none> {
  %1 = lit.struct.create() : () -> !lit.struct<@Error>
  %2 = kgen.variant.create %1, 0 : <@Error, none>
  // CHECK: kgen.return %1 : !kgen.variant<@Error, none>
  lit.error_return %2 : !kgen.variant<@Error, none>
}

// CHECK-LABEL: kgen.generator @return_raise_or
// CHECK-SAME: -> !kgen.variant<@Error, none>
lit.func @return_raise_or(%cond: i1, %err: !lit.struct<@Error>) -> !kgen.variant<@Error, none> {
  // CHECK-NEXT: hlcf.if %arg0
  hlcf.elif {
    hlcf.elif.yield %cond : i1
  } then {
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
// CHECK-SAME: (%arg0: !lit.ref<index, imm *[0,0]> inout) throws ->
lit.func @removeMetadata[imm a](%arg0: !lit.ref<index, imm a> inout) throws -> !kgen.variant<@Error, index> {
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
    // CHECK-LABEL: kgen.generator @"module::Adder::__add__"<size>(%arg0: !lit.struct<#Adder <size>>)
    // CHECK-NEXT:    kgen.call @"module::test"() : () -> ()
    lit.func @__add__(%self: !lit.struct<@module::@Adder<size>>)  {
      lit.call @module::@test() : !lit.signature<() -> ()>
      kgen.return
    }
  }

  // CHECK-LABEL: lit.struct.decl @"module::Adder"<size> {
}

// CHECK-LABEL: kgen.generator @caller(%arg0: !lit.struct<#Adder <10>>)
lit.func @caller(%ref: !lit.struct<@module::@Adder<10>>)  {
  // CHECK: kgen.call @"module::Adder::__add__"
  kgen.call @module::@Adder::@__add__<10>(%ref) : !lit.signature<("self": !lit.struct<@module::@Adder<10>>) -> ()>
  kgen.return
}

// -----

// CHECK-NOT: lit.package
lit.package @package {
  // CHECK-NOT: lit.file_module
  lit.file_module @module {
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

// -----

// COM: Ensure the linkage name is respected when it could conflict.

// CHECK: kgen.generator export @main
lit.package @main {
  lit.file_module @main {
    lit.func export @main() attributes {linkageName = "main"} {
      kgen.return
    }
  }
}


//===----------------------------------------------------------------------===//
// Implicit lifetimes.
//===----------------------------------------------------------------------===//

// -----

// Verify that the lifetimes get correctly removed and the IR is correct.

!Mem = !lit.struct<@Mem>
lit.struct.decl @Mem   {
  lit.func @__init__[mut a](%self: !lit.ref<!Mem, mut a> init_self, |) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

// CHECK-LABEL: kgen.generator @getThing
// CHECK-SAME:(%arg0: !lit.ref<@Mem, mut *[0,0]> byref_result)
lit.func @getThing[mut abc](%res: !lit.ref<!Mem, mut abc> byref_result, |) -> !kgen.none {
  // CHECK-NEXT: kgen.param.declare abc: lifetime<1> = <#lit.lifetime>
  // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : !lit.ref<@Mem, mut *[0,0]> to !lit.ref<@Mem, mut abc>

  // CHECK-NEXT: kgen.param.declare.region localTest = (%arg1: !lit.ref<@Mem, mut *[0,0]> byref_result) capturing
  lit.func localTest[mut lt](%__result__[__result__]: !lit.ref<!Mem, mut lt> byref_result, |) capturing -> !kgen.none {
    // CHECK-NEXT: kgen.param.declare lt: lifetime
    // CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %arg1 : !lit.ref<@Mem, mut *[0,0]> to !lit.ref<@Mem, mut lt>
    %1 = lit.call @Mem::@__init__[mut lt](%__result__) : !lit.signature<[1]("self": !lit.ref<!Mem, mut *[0,0]> init_self, |) -> !kgen.none>
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
  // CHECK: }
  // CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0
  // CHECK-NEXT: %2 = kgen.call_param[(!lit.ref<@Mem, mut *[0,0]> byref_result) capturing -> !kgen.none: localTest](%1)
  %0 = lit.call[!lit.signature<[1]("__result__": !lit.ref<!Mem, mut *[0,0]> byref_result, |) capturing -> !kgen.none>: localTest][mut abc](%res)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}


// CHECK-LABEL: kgen.generator @callThing
// CHECK-SAME: (%arg0: !lit.ref<@Mem, mut *[0,0]> byref_result)
lit.func @callThing[mut lt](%__result__: !lit.ref<!Mem, mut lt> byref_result, |) -> !kgen.none attributes {isParametric, sourceName = "callThing", specialFnKind = 0 : i8} {
  // CHECK-NEXT: kgen.param.declare lt: lifetime<1> = <#lit.lifetime>
  // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : !lit.ref<@Mem, mut *[0,0]> to !lit.ref<@Mem, mut lt>
  // CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0 : !lit.ref<@Mem, mut lt> to !lit.ref<@Mem, mut *[0,0]>
  // CHECK-NEXT: kgen.call @getThing(%1)
  %0 = lit.call @getThing[mut lt](%__result__) : !lit.signature<[1]("res": !lit.ref<!Mem, mut *[0,0]> byref_result, |) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @testLifetimeOf2
// Verify that we remap the returns as well as the operands.
lit.func @testLifetimeOf2[imm *"a`"](%a: !lit.ref<!Mem, imm *"a`"> borrow_in_mem) -> !lit.ref<!Mem, imm *"a`">{
  // CHECK-NEXT: kgen.param.declare *"a`"
  // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : !lit.ref<@Mem, imm *[0,0]> to !lit.ref<@Mem, imm *"a`">
  // CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %0 : !lit.ref<@Mem, imm *"a`"> to !lit.ref<@Mem, imm *[0,0]>
  // CHECK-NEXT: kgen.return %1
  kgen.return %a : !lit.ref<!Mem, imm *"a`">
}

//===----------------------------------------------------------------------===//
// Ownership
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @ownership_ops
lit.func @ownership_ops[mut lt](%a: !lit.ref<index, mut lt>) {
  // CHECK-NOT: lit.ownership.
  lit.ownership.mark_initialized %a : !lit.ref<index, mut lt>
  lit.ownership.use %a : !lit.ref<index, mut lt>
  lit.ownership.mark_destroyed %a : !lit.ref<index, mut lt>
  kgen.return
}
