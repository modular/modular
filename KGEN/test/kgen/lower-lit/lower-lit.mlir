// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

lit.fn @callee[imm a, mut b]() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @calls
lit.fn @calls<f: !lit.generator<[2]() -> ()>>[imm a, mut b](%arg0: !lit.generator<[2]() -> ()>) {
  // CHECK: kgen.call @callee() : () -> ()
  lit.call @callee[imm a, mut b]() : !lit.generator<[2]() -> ()>
  // CHECK: kgen.call_param[() -> (): f]()
  lit.call[!lit.generator<[2]() -> ()>: f][imm a, mut b]()
  // CHECK: kgen.call_indirect %arg0() : () -> ()
  lit.call_indirect %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>
  kgen.return
}

lit.fn @async_fn_throws(%err: !lit.ref<index, mut #lit.any.origin> byref_error, %res: !lit.ref<index, mut #lit.any.origin> byref_result) throws|async {
  kgen.return
}

// CHECK-LABEL: kgen.generator @async_call
lit.fn @async_call[imm a, mut b]() async {
  // CHECK: co.invoke[() async -> (): @async_call]()
  lit.async.call[!lit.generator<[2]() async -> ()>: @async_call][imm a, mut b]()
  // CHECK: co.invoke[(!kgen.pointer<index> byref_error, !kgen.pointer<index> byref_result) throws|async -> (): @async_fn_throws]()
  lit.async.call[!lit.generator<("err": !lit.ref<index, mut #lit.any.origin> byref_error, "res": !lit.ref<index, mut #lit.any.origin> byref_result) throws|async -> ()>: @async_fn_throws]()
  kgen.return
}

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT: }
lit.fn @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @varDecl
// CHECK-SAME:  (%[[ARG0:.*]]: index) -> index
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    kgen.return %[[ARG0]] : index
// CHECK-NEXT:  }

lit.fn @varDecl(%arg0: index) -> index {
  %a = lit.var.decl "a" var : !lit.ref<index, mut *"life">
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.generator @varDecl2
// CHECK-SAME:  (%[[ARG0:.*]]: index)
// CHECK-NEXT: %0 = pop.stack_allocation 1 x index
// CHECK-NEXT: kgen.return
lit.fn @varDecl2(%arg0: index) {
  %a = lit.var.decl "a" var : !lit.ref<index, mut alife>
  kgen.return
}

lit.fn @decorator() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @decorated_fn
lit.fn @decorated_fn()
  // CHECK-NEXT: decorators <:() -> () @decorator>
  decorators<:!lit.generator<() -> ()> @decorator> {
  kgen.return
}

// CHECK-LABEL: @generic_types_retain_convention
lit.fn @generic_types_retain_convention<T: type>[imm a](
  // CHECK: %arg0: !kgen.param<T>,
  // CHECK: %arg1: !kgen.pointer<T> mut,
  // CHECK: %arg2: !kgen.param<T> owned,
  // CHECK: %arg3: index,
  // CHECK: %arg4: !kgen.pointer<index> owned
  %p: !kgen.param<T>,
  %q: !lit.ref<T, imm a> mut,
  %r: !kgen.param<T> owned,
  %s1: index,
  %s2: !kgen.pointer<index> owned
){
  kgen.return
}

lit.fn @generic_callee<T: type>(%p: !kgen.param<T>){
  kgen.return
}

// CHECK-LABEL: @call_generic
lit.fn @call_generic(%p: index) {
  // CHECK: kgen.call @generic_callee<:type index>({{.*}}) : (index) -> ()
  kgen.call @generic_callee<:type index>(%p) : !lit.generator<("p": index) -> ()>
  kgen.return
}


//===----------------------------------------------------------------------===//
// Nested Functions
//===----------------------------------------------------------------------===//

lit.struct.decl @StructWithNestedFn<a_param> {
  // CHECK-LABEL: kgen.generator @"StructWithNestedFn::topLevelFunction"<a_param, b_param>() -> index
  lit.fn @topLevelFunction<b_param>() -> index {
    // CHECK: kgen.param.declare.region nestedFunction = () -> index
    lit.fn nestedFunction() -> index {
      kgen.unreachable
    }
    // CHECK: kgen.param.declare b: () -> index = <nestedFunction>
    kgen.param.declare b: !lit.generator<() -> index> = <nestedFunction>

    // CHECK: kgen.param.declare.region paramNestedFunc = <c_param>()
    lit.fn paramNestedFunc<c_param>() {
      kgen.return
    }
    // CHECK: kgen.param.declare c: () -> () = <bind_params(:<index>() -> () paramNestedFunc, 2)>
    kgen.param.declare c: !lit.generator<() -> ()> = <bind_params(:!lit.generator<<"c_param": index>() -> ()> paramNestedFunc, 2)>

    %idx0_0 = index.constant 0
    kgen.return %idx0_0 : index
  }
}

// CHECK-LABEL: kgen.struct.generator @StructWithNestedFn<a_param>

// CHECK-LABEL: kgen.generator @topFunc
lit.fn @topFunc() {
  // CHECK: kgen.param.declare.region midFunc
  lit.fn midFunc() {
    // CHECK: kgen.param.declare.region botFunc
    lit.fn botFunc() {
      kgen.return
    }
    // CHECK: declare bot: () -> () = <botFunc>
    kgen.param.declare bot: !lit.generator<() -> ()> = <botFunc>
    kgen.return
  }
  // CHECK: declare mid: () -> () = <midFunc>
  kgen.param.declare mid: !lit.generator<() -> ()> = <midFunc>
  kgen.return
}

//===----------------------------------------------------------------------===//
// Imports
//===----------------------------------------------------------------------===//

// -----

// CHECK-NOT: lit.unresolved_import
lit.file_module @nested_imports {
  lit.unresolved_import @foobar as @foo

  lit.fn @func() {
    lit.unresolved_import @foobar as @foo
    kgen.return
  }
}

//===----------------------------------------------------------------------===//
// Structs
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @Adder<size> {
  // CHECK-LABEL: kgen.generator @"Adder::__add__"<size>(%arg0: !kgen.struct<() memoryOnly>)
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.fn @__add__(%self: !lit.struct<@Adder<size>>)  {
    %0 = lit.var.decl "a" var : !lit.ref<index, mut *"life">
    %one = index.constant 1
    lit.ref.store %one, %0 : !lit.ref<index, mut *"life">
    kgen.return
  }
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"

// CHECK-LABEL: kgen.struct.generator @A
lit.struct.decl @A {
  lit.fn @foo(%self: !lit.struct<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @"B::foo"
// CHECK-NEXT: call_param[(!kgen.struct<() memoryOnly>) -> (): @"A::foo"]

// CHECK-LABEL: kgen.struct.generator @B
lit.struct.decl @B {
  lit.fn @foo(%self: !lit.struct<@B>, %a: !lit.struct<@A>) {
    kgen.call_param[!lit.generator<("self": !lit.struct<@A>) -> ()>: @A::@foo](%a)
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @main
lit.fn @main(%a: !lit.struct<@A>, %b: !lit.struct<@B>) {
  // CHECK-NEXT: call_param[(!kgen.struct<() memoryOnly>, !kgen.struct<() memoryOnly>) -> (): @"B::foo"]
  kgen.call_param[!lit.generator<("self": !lit.struct<@B>, "a": !lit.struct<@A>) -> ()>: @B::@foo](%b, %a)
  // CHECK-NEXT: constant: (!kgen.struct<() memoryOnly>) -> () = <@"A::foo">
  %0 = kgen.param.constant: !lit.generator<("self": !lit.struct<@A>) -> ()> = <@A::@foo>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"<N, M>

// CHECK-LABEL: kgen.struct.generator @A<N>
lit.struct.decl @A<N> {
  lit.fn @foo<M>(%self: !lit.struct<@A<N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: kgen.generator @main
lit.fn @main(%a: !lit.struct<@A<1>>) {
  // CHECK-NEXT: call_param[(!kgen.struct<() memoryOnly>) -> index: @"A::foo"<1, 2>]
  %0 = kgen.call_param[!lit.generator<("self": !lit.struct<@A<1>>) -> index>: @A::@foo<1, 2>](%a)
  kgen.return
}

// -----

lit.struct.decl @A {
}

// CHECK: kgen.generator @rhslitdeclref_no_params(%arg0: !kgen.struct<() memoryOnly>)
lit.fn @rhslitdeclref_no_params(%x: !lit.struct<@A>) {
  kgen.return
}

// -----

lit.struct.decl @A<b, c> {
}

// CHECK: kgen.generator @rhslitdeclref_params(%arg0: !kgen.struct<() memoryOnly>)
lit.fn @rhslitdeclref_params(%x: !lit.struct<@A<10, 11>>) {
  kgen.return
}

// -----

lit.struct.decl @A {
  lit.fn @B() {
    kgen.return
  }
}

// CHECK-LABEL: @callIt
lit.fn @callIt() {
  // CHECK-NEXT: kgen.call @"A::B"
  lit.call @A::@B() : !lit.generator<() -> ()>
  kgen.return
}

// -----

// CHECK-NOT: lit.alias.decl
lit.alias.decl A = <1>
lit.struct.decl @foo {
  // CHECK-NOT: lit.alias.decl
  lit.alias.decl B = <2>
 // CHECK-LABEL:  @"foo::f"() -> index
  lit.fn @f() -> index {
    // CHECK-NOT: kgen.param.declare
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

lit.fn @throwing_func(%1: !lit.struct<@Error>) throws -> !kgen.variant<@Error, none> {
  %2 = kgen.variant.create %1, 0 : <@Error, none>
  // CHECK: kgen.return %0 : !kgen.variant<struct<() memoryOnly>, none>
  lit.error_return %2 : !kgen.variant<@Error, none>
}

// CHECK-LABEL: kgen.generator @return_raise_or
// CHECK-SAME: -> !kgen.variant<struct<() memoryOnly>, none>
lit.fn @return_raise_or(%cond: i1, %err: !lit.struct<@Error>) -> !kgen.variant<@Error, none> {
  // CHECK-NEXT: hlcf.if %arg0
  hlcf.elif {
    hlcf.elif.yield %cond
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
// CHECK-SAME: (%arg0:  !kgen.pointer<index> mut) throws ->
lit.fn @removeMetadata[imm a](%arg0: !lit.ref<index, imm a> mut) throws -> !kgen.variant<@Error, index> {
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
  lit.globalvar.ref @foo : <index, mut #lit.any.origin>
  // CHECK-NEXT: %1 = kgen.global.address @baz
  lit.globalvar.ref @baz : <index, mut #lit.any.origin>
  // CHECK-NEXT: kgen.return
}, {
}
// CHECK: kgen.global @bar : index [{{.*}}](2)

// CHECK: kgen.global @baz : index [{{.*}}](1)
lit.globalvar.decl @baz : index {
  lit.globalvar.ref @foo : <index, mut #lit.any.origin>
}, {
}

// CHECK: kgen.global @boo : index [{{.*}}](3)
lit.globalvar.decl @boo : index {
  lit.globalvar.ref @bar : <index, mut #lit.any.origin>
  lit.globalvar.ref @baz : <index, mut #lit.any.origin>
}, {
}

// -----

lit.file_module @module {
  // CHECK: kgen.global export @foo : index
  lit.globalvar.decl export @exported : index attributes {linkageName = "foo"} {}, {}

  // CHECK-LABEL: kgen.generator @"module::ref_exported"
  lit.fn @ref_exported() {
    // CHECK-NEXT: kgen.global.address @foo : <index>
    %0 = lit.globalvar.ref @module::@exported : <index, mut #lit.any.origin>
    kgen.return
  }
}

// -----

// CHECK: kgen.generator @"(ctor_fn)self"
lit.globalvar.decl @self : index {
  // CHECK-NEXT: kgen.global.address @self
  lit.globalvar.ref @self : <index, mut #lit.any.origin>
}, {
  lit.globalvar.ref @self : <index, mut #lit.any.origin>
}

// -----

//===----------------------------------------------------------------------===//
// Modules
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.file_module

lit.file_module @module {
  // CHECK: kgen.generator @"module::test"()
  lit.fn @test()  {
    kgen.return
  }

  lit.struct.decl @Adder<size> {
    // CHECK-LABEL: kgen.generator @"module::Adder::__add__"<size>(%arg0: !kgen.struct<() memoryOnly>)
    // CHECK-NEXT:    kgen.call @"module::test"() : () -> ()
    lit.fn @__add__(%self: !lit.struct<@module::@Adder<size>>)  {
      lit.call @module::@test() : !lit.generator<() -> ()>
      kgen.return
    }
  }

  // CHECK-LABEL: kgen.struct.generator @"module::Adder"<size>
}

// CHECK-LABEL: kgen.generator @caller(%arg0: !kgen.struct<() memoryOnly>)
lit.fn @caller(%ref: !lit.struct<@module::@Adder<10>>)  {
  // CHECK: kgen.call @"module::Adder::__add__"
  kgen.call @module::@Adder::@__add__<10>(%ref) : !lit.generator<("self": !lit.struct<@module::@Adder<10>>) -> ()>
  kgen.return
}

// -----

// CHECK-NOT: lit.package
lit.package @package {
  // CHECK-NOT: lit.file_module
  lit.file_module @module {
    // CHECK: kgen.generator export @"package::module::foo"()
    lit.fn export @foo() {
      kgen.return
    }
  }
}

// -----

lit.file_module @module {
  // CHECK-NOT: lit.alias.decl
  lit.alias.decl A = <42>
}

// CHECK: kgen.generator @metadata
// CHECK-SAME{LITERAL}: LLVMArgMetadataArray = [[], ["llvm.someattr", 2 : index]]
// CHECK-SAME: LLVMMetadataArray = ["llvm.someattr",  3 : index]
lit.fn @metadata(%a: i32, %b: i32) attributes {
  LLVMArgMetadataArray = [[], ["llvm.someattr", 2 : index]],
  LLVMMetadataArray = ["llvm.someattr", 3 : index]
} {
  // CHECK: kgen.param.declare.region metadataNested
  lit.fn metadataNested(%c: i32, %d: i32) attributes {
    LLVMArgMetadataArray = [[], ["llvm.someattr", 4 : index]],
    LLVMMetadataArray = ["llvm.someattr",  5 : index]
  } {
    // CHECK-NEXT: kgen.return
    kgen.return
  // CHECK-NEXT{LITERAL}: LLVMArgMetadataArray = [[], ["llvm.someattr", 4 : index]]
  // CHECK-SAME: LLVMMetadataArray = ["llvm.someattr", 5 : index]
  }
  kgen.return
}

// -----

// COM: Ensure the linkage name is respected when it could conflict.

// CHECK: kgen.generator export @main
lit.package @main {
  lit.file_module @main {
    lit.fn export @main() attributes {linkageName = "main"} {
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
  lit.fn @__init__[mut a](%self: !lit.ref<!Mem, mut a> byref_result, |) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

// CHECK-LABEL: kgen.generator @getThing
// CHECK-SAME:(%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
lit.fn @getThing[mut abc](%res: !lit.ref<!Mem, mut abc> byref_result, |) -> !kgen.none {
  // CHECK-NEXT: kgen.param.declare.region localTest = (%arg1: !kgen.pointer<struct<() memoryOnly>> byref_result) capturing
  lit.fn localTest[mut lt](%__result__[__result__]: !lit.ref<!Mem, mut lt> byref_result, |) capturing -> !kgen.none {
    // CHECK-NEXT: call @"Mem::__init__"(%arg1)
    %1 = lit.call @Mem::@__init__[mut lt](%__result__) : !lit.generator<[1]("self": !lit.ref<!Mem, mut *[0,0]> byref_result, |) -> !kgen.none>
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
  // CHECK: }
  // CHECK-NEXT: kgen.call_param[(!kgen.pointer<struct<() memoryOnly>> byref_result) capturing -> !kgen.none: localTest](%arg0)
  %0 = lit.call[!lit.generator<[1]("__result__": !lit.ref<!Mem, mut *[0,0]> byref_result, |) capturing -> !kgen.none>: localTest][mut abc](%res)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}


// CHECK-LABEL: kgen.generator @callThing
// CHECK-SAME: (%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
lit.fn @callThing[mut lt](%__result__: !lit.ref<!Mem, mut lt> byref_result, |) -> !kgen.none attributes {isParametric, sourceName = "callThing", specialFnKind = 0 : i8} {
  // CHECK-NEXT: kgen.call @getThing(%arg0)
  %0 = lit.call @getThing[mut lt](%__result__) : !lit.generator<[1]("res": !lit.ref<!Mem, mut *[0,0]> byref_result, |) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @testLifetimeOf2
// Verify that we remap the returns as well as the operands.
lit.fn @testLifetimeOf2[imm *"a`"](%a: !lit.ref<!Mem, imm *"a`"> read_mem) -> !lit.ref<!Mem, imm *"a`">{
  // CHECK-NEXT: kgen.return %arg0
  kgen.return %a : !lit.ref<!Mem, imm *"a`">
}

// CHECK-LABEL: kgen.generator @callLifetimes
// CHECK-SAME: (%arg0: !kgen.pointer<index>) -> !kgen.pointer<index>
lit.fn @callLifetimes[mut lt](%arg0[*""]: !lit.ref<index, mut lt>) -> !lit.ref<index, mut lt> {
  // CHECK: kgen.call @callLifetimes(%arg0) : (!kgen.pointer<index>) -> !kgen.pointer<index>
  %0 = lit.call @callLifetimes[mut lt](%arg0) : !lit.generator<[1](!lit.ref<index, mut *[0,0]>) -> !lit.ref<index, mut *[0,0]>>
  kgen.return %0 : !lit.ref<index, mut lt>
}


// This should drop the explicit origin parameters since they are singletons.

// CHECK-LABEL: kgen.generator @takes_life_explicit<ismut: i1, size, val: simd<size, f32>>
// CHECK-SAME: (%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
lit.fn @takes_life_explicit<ismut: i1, life: !lit.origin<ismut>, size: index, val: !pop.simd<size, f32>>
                    (%ref: !lit.ref<!Mem, mut=ismut, life> byref_result, |) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_takes_life_explicit
// CHECK-SAME: <val: simd<4, f32>>(%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
lit.fn @call_takes_life_explicit<val: !pop.simd<4, f32>>[mut lt](%__result__: !lit.ref<!Mem, mut lt> byref_result, |) {
  // CHECK-NEXT: kgen.call @takes_life_explicit<:i1 1, 4, :simd<4, f32> val>(%arg0)
  // CHECK-SAME: : (!kgen.pointer<struct<() memoryOnly>> byref_result) -> ()
  lit.call @takes_life_explicit<:i1 1, :!lit.origin<1> lt, :index 4, :!pop.simd<4, f32> val>(%__result__)
      : !lit.generator<("ref": !lit.ref<!Mem, mut lt> byref_result, |) -> ()>
  kgen.return
}

// -----

!Int = !lit.struct<@Int>
#IndexList = #lit<symbol@IndexList>
lit.struct.decl @Int {
  lit.struct.field value : index
}

lit.struct.decl @IndexList<size: !Int> {
  lit.fn @getitem(%self[*""]: !lit.struct<@IndexList<:!Int size>>) -> !Int {
    kgen.unreachable
  }
}

// CHECK-LABEL: kgen.generator @paramReplacement
// CHECK-SAME: callee: (!kgen.struct<() memoryOnly>) -> ()>
lit.fn @paramReplacement<
    _1: !Int,
    _2: @IndexList<:!Int _1>,
    callee: !lit.generator<[1](!lit.struct<#IndexList <:!Int apply(:!lit.generator<(!lit.struct<#IndexList <:!Int _1>>) -> !Int> @IndexList::@getitem<:!Int _1>, _2)>>) -> ()>>(){
  kgen.unreachable
}

// -----

//===----------------------------------------------------------------------===//
// Ownership
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @ownership_ops
lit.fn @ownership_ops[mut lt](%a: !lit.ref<index, mut lt>) {
  // CHECK-NOT: lit.ownership.
  lit.ownership.mark_initialized %a : !lit.ref<index, mut lt>
  lit.ownership.use %a : !lit.ref<index, mut lt>
  lit.ownership.mark_destroyed %a : !lit.ref<index, mut lt>
  kgen.return
}

//===----------------------------------------------------------------------===//
// Singleton Struct Types.
//===----------------------------------------------------------------------===//

lit.struct.decl @EmptyStruct {}

// CHECK-LABEL: kgen.generator @expect_always_empty_struct()
lit.fn @expect_always_empty_struct<es: !lit.struct<@EmptyStruct>>() {
  kgen.return
}

lit.fn @expect_parametric_empty_struct<t: type, s: !kgen.param<t>>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_using_empty_struct<alwaysFn: <index>() -> (), paramFn: <type, *(0,0)>() -> ()>()
lit.fn @call_using_empty_struct<es: !lit.struct<@EmptyStruct>, alwaysFn: <index, @EmptyStruct>() -> (), paramFn: <type, *(0,0)>() -> ()>() {
  // CHECK-NEXT: kgen.call @expect_always_empty_struct() : () -> ()
  lit.call @expect_always_empty_struct<:!lit.struct<@EmptyStruct> es>() : !lit.generator<() -> ()>
  // CHECK-NEXT: kgen.call @expect_parametric_empty_struct<:type {{.*}}, :struct<() memoryOnly> {  }>()
  lit.call @expect_parametric_empty_struct<:type #kgen.type<!lit.struct<@EmptyStruct>>, :!lit.struct<@EmptyStruct> es>() : !lit.generator<() -> ()>
  // CHECK-NEXT: <bind_params(:<index>() -> () alwaysFn, 1)>
  kgen.param.declare alwaysFn2: !lit.generator<() -> ()> = <bind_params(:<index, @EmptyStruct>() -> () alwaysFn, 1, es)>
  // CHECK-NEXT: <bind_params(:<type, *(0,0)>() -> () paramFn, {{.*}}, {  })>
  kgen.param.declare paramFn2: !lit.generator<() -> ()> = <bind_params(:<type, *(0,0)>() -> () paramFn, #kgen.type<!lit.struct<@EmptyStruct>>, es)>
  kgen.return
}
