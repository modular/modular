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
  lit.call [!lit.generator<[2]() -> ()>: f][imm a, mut b]()
  // CHECK: kgen.call_indirect %arg0() : () -> ()
  lit.call_indirect %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>

  // CHECK: kgen.call tail @callee() : () -> ()
  lit.call tail @callee[imm a, mut b]() : !lit.generator<[2]() -> ()>
  // CHECK: kgen.call_indirect musttail %arg0() : () -> ()
  lit.call_indirect musttail %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>
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

// Check removing metadata and singleton types from generator types.

lit.struct.decl @EmptyStruct {}

lit.fn @empty_fn<t: !lit.struct<@EmptyStruct>>(%arg0: index) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @removeGenMetadata
lit.fn @removeGenMetadata() {
  // CHECK-NEXT: <index, index>index = <#kgen.gen<add(*(0,0), *(0,1))>>
  kgen.param.declare test: !lit.generator<<"a": index, "b": index> index> = <#kgen.gen<add(*(0,0), *(0,1))>>
  // CHECK-NEXT: struct<() memoryOnly> = <{  }>
  kgen.param.declare test2: !lit.generator<<"t": !lit.struct<@EmptyStruct>> !lit.struct<@EmptyStruct>> = <#kgen.gen<*(0,0)>>
  // CHECK-NEXT: (index) -> () = <@empty_fn>
  kgen.param.declare test3: !lit.generator<<"t": !lit.struct<@EmptyStruct>> ("arg0":index) -> ()> = <@empty_fn>
  // CHECK-NEXT: <index>struct<(struct<() memoryOnly>, index)> = <#kgen.gen<{ { }, *(0,0) }>>
  kgen.param.declare test4: !lit.generator<<"x": !lit.struct<@EmptyStruct>, "y": index = 5> !kgen.struct<(!lit.struct<@EmptyStruct>, index)>> = <#kgen.gen<#kgen.struct<*(0,0), *(0,1)>>>
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
// Traits
//===----------------------------------------------------------------------===//

lit.trait.decl @RetZero {
  // CHECK-LABEL: kgen.generator @"RetZero::return_zero"
  lit.fn @return_zero() -> index {
    %idx0_0 = index.constant 0
    kgen.return %idx0_0 : index
  }
}

// -----

lit.trait.decl @NestedParams<A> {
  // CHECK-LABEL: kgen.generator @"NestedParams::nested_params"<A, B>
  lit.fn @nested_params<B>() -> index {
    %idx0_0 = index.constant 0
    kgen.return %idx0_0 : index
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
// Modules
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.file_module

lit.file_module @module {
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
  // CHECK: kgen.generator @"module::test"()
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

// COM: Ensure the linkage name is passed through on an exported function.

// CHECK: kgen.generator export @"main::main::main"
// CHECK-SAME: linkageName = #kgen.linkage_name<"main" : !kgen.string, false>
lit.package @main {
  lit.file_module @main {
    lit.fn export @main() attributes {linkageName = #kgen.linkage_name<"main" : !kgen.string, false>} {
      kgen.return
    }
  }
}

// -----

// COM: Ensure that linkageName (static string) is passed through without
// COM: renaming the symbol.

// CHECK: kgen.generator export @"pkg::mod::my_fn"
// CHECK-SAME: linkageName = #kgen.linkage_name<"my_export" : !kgen.string, false>
lit.package @pkg {
  lit.file_module @mod {
    lit.fn export @my_fn() attributes {linkageName = #kgen.linkage_name<"my_export" : !kgen.string, false>} {
      kgen.return
    }
  }
}

// -----

// COM: Ensure that linkageName on a non-export function passes through.

// CHECK: kgen.generator @"pkg3::mod3::orig_name"
// CHECK-SAME: linkageName = #kgen.linkage_name<"my_link_name" : !kgen.string, false>
lit.package @pkg3 {
  lit.file_module @mod3 {
    lit.fn @orig_name() attributes {linkageName = #kgen.linkage_name<"my_link_name" : !kgen.string, false>} {
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
  %0 = lit.call [!lit.generator<[1]("__result__": !lit.ref<!Mem, mut *[0,0]> byref_result, |) capturing -> !kgen.none>: localTest][mut abc](%res)
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}


// CHECK-LABEL: kgen.generator @callThing
// CHECK-SAME: (%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
// CHECK-SAME: sourceName = "callThing"
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
lit.fn @takes_life_explicit<ismut: i1, life: !lit.origin<ismut>, size: index, val: !kgen.simd<size, f32>>
                    (%ref: !lit.ref<!Mem, mut=ismut, life> byref_result, |) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_takes_life_explicit
// CHECK-SAME: <val: simd<4, f32>>(%arg0: !kgen.pointer<struct<() memoryOnly>> byref_result)
lit.fn @call_takes_life_explicit<val: !kgen.simd<4, f32>>[mut lt](%__result__: !lit.ref<!Mem, mut lt> byref_result, |) {
  // CHECK-NEXT: kgen.call @takes_life_explicit<:i1 1, 4, :simd<4, f32> val>(%arg0)
  // CHECK-SAME: : (!kgen.pointer<struct<() memoryOnly>> byref_result) -> ()
  lit.call @takes_life_explicit<:i1 1, :!lit.origin<1> lt, :index 4, :!kgen.simd<4, f32> val>(%__result__)
      : !lit.generator<("ref": !lit.ref<!Mem, mut lt> byref_result, |) -> ()>
  kgen.return
}

// -----

!Int = !lit.struct<@Int>
#IndexList = #lit<symbol@IndexList>
lit.struct.decl @Int {
  lit.struct.field value : index
}

lit.struct.decl @IndexList<life: !lit.origin<1>, size: !Int> {
  lit.fn @getitem(%self[*""]: !lit.struct<@IndexList<:!lit.origin<1> life, :!Int size>>) -> !Int {
    kgen.unreachable
  }
}

// COM: Origin parameters are singleton values and must be dropped from
// COM: TypeGeneratorRefAttr bindings during LowerLIT.
// CHECK-LABEL: kgen.generator @"IndexList::getitem"<size: struct<(index) memoryOnly>>(
// CHECK-LABEL: kgen.generator @paramReplacement<_1: struct<(index) memoryOnly>, callee: (!kgen.struct<() memoryOnly>) -> ()>()
lit.fn @paramReplacement<
    _1: !Int,
    _2: @IndexList<:!lit.origin<1> lt, :!Int _1>,
    callee: !lit.generator<[1](!lit.struct<#IndexList <:!lit.origin<1> lt, :!Int apply(:!lit.generator<(!lit.struct<#IndexList <:!lit.origin<1> lt, :!Int _1>>) -> !Int> @IndexList::@getitem<:!lit.origin<1> lt, :!Int _1>, _2)>>) -> ()>>[mut lt]() {
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
lit.fn @call_using_empty_struct<es: !lit.struct<@EmptyStruct>, alwaysFn: !lit.generator<<index, @EmptyStruct>() -> ()>, paramFn: !lit.generator<<type, !kgen.param<*(0,0)>>() -> ()>>() {
  // CHECK-NEXT: kgen.call @expect_always_empty_struct() : () -> ()
  lit.call @expect_always_empty_struct<:!lit.struct<@EmptyStruct> es>() : !lit.generator<() -> ()>
  // CHECK-NEXT: kgen.call @expect_parametric_empty_struct<:type {{.*}}, :struct<() memoryOnly> {  }>()
  lit.call @expect_parametric_empty_struct<:type #kgen.type<!lit.struct<@EmptyStruct>>, :!lit.struct<@EmptyStruct> es>() : !lit.generator<() -> ()>
  // CHECK-NEXT: <bind_params(:<index>() -> () alwaysFn, 1)>
  kgen.param.declare alwaysFn2: !lit.generator<() -> ()> = <bind_params(:!lit.generator<<index, @EmptyStruct>() -> ()> alwaysFn, :index 1, :!lit.struct<@EmptyStruct> es)>
  // CHECK-NEXT: <bind_params(:<type, *(0,0)>() -> () paramFn, {{.*}}, :struct<() memoryOnly> {  })>
  kgen.param.declare paramFn2: !lit.generator<() -> ()> = <bind_params(:!lit.generator<<type, !kgen.param<*(0,0)>>() -> ()> paramFn, #kgen.type<!lit.struct<@EmptyStruct>>, :!lit.struct<@EmptyStruct> es)>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Closures.
//===----------------------------------------------------------------------===//

!Closure = !lit.trait<@Closure>
!String = !lit.struct<@String>
!Impl = !kgen.closure<@make_closure, "foo" nonescaping>

#Impl1 = #kgen.type<!Impl> : !Closure

lit.trait.decl @Closure<?, SELF: !Closure> {
  lit.fn @"__call__"[imm O](%self: !lit.ref<:!Closure SELF, imm O> read_mem, %y: index) -> index {
    kgen.unreachable
  }
}

// CHECK-label: kgen.generator @make_closure
lit.fn @make_closure[imm Y, imm Z](%y: !lit.ref<!String, imm Y> owned_in_mem, %x:index, %z: !lit.ref<!String, imm Z> owned_in_mem) -> !kgen.none {
  // CHECK:     kgen.closure.init(%arg0, %arg1, %arg2[@"String::__copyinit__", @"String::__moveinit__", @"String::__del__"])(%arg3: index) -> index {
  // CHECK-NEXT:    kgen.return %arg1 : index
  // CHECK-NEXT: } : (!kgen.pointer<struct<() memoryOnly>>, index, !kgen.pointer<struct<() memoryOnly>>)
  // CHECK-SAME: , !kgen.pointer<!kgen.closure<@make_closure, "foo" nonescaping>>
  %impl = lit.closure.init[#Impl1](%y[ref: imm Y], %x, %z[@String::@__copyinit__ !lit.generator<[2]("existing": !lit.ref<!String, imm *[0,1]> read_mem, "self": !lit.ref<!String, mut *[0,0]> byref_result) -> !kgen.none>,
                                                  @String::@__moveinit__ !lit.generator<[2]("existing": !lit.ref<!String, imm *[0,1]> read_mem, "self": !lit.ref<!String, mut *[0,0]> byref_result) -> !kgen.none>,
                                                  @String::@__del__ !lit.generator<[1]("self": !lit.ref<!String, mut *[0,0]> owned_in_mem) -> !kgen.none>])(%y2: index) -> index {
   kgen.return %x : index
  } : (!lit.ref<!String, imm Y>, index, !lit.ref<!String, imm Z>), !lit.ref<!kgen.closure<@make_closure, "foo" nonescaping>, mut C>
  %2 = lit.call @direct[mut C]<:!Closure #Impl1>(%impl, %x) : !lit.generator<[1]("c":!lit.ref<:!Closure #Impl1, mut *[0,0]> read_mem, "x": index) -> !kgen.none>

   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}

lit.fn @direct<CT: !Closure>[mut Origin0](%c: !lit.ref<:!Closure CT, mut Origin0> read_mem, %x: index) -> !kgen.none {
   %0 = lit.call [!lit.generator<[1]("self": !lit.ref<:!Closure CT, imm *[0,0]> read_mem, "y": index) -> index>:
        #kgen.get_witness<:!Closure CT, "Closure", "__call__">][mut Origin0](%c, %x)
   lit.end_fn
}

lit.struct.decl @String {
    lit.fn @__copyinit__[mut E1, imm E2](%existing: !lit.ref<!String, imm E2> read_mem, %self: !lit.ref<!String, mut E1> byref_result) -> !kgen.none {
      %none = kgen.param.constant: none = <#kgen.none>
      kgen.return %none : !kgen.none
    }
    lit.fn @__moveinit__[mut E1, imm E2](%existing: !lit.ref<!String, imm E2> read_mem, %self: !lit.ref<!String, mut E1> byref_result) -> !kgen.none {
      %none = kgen.param.constant: none = <#kgen.none>
      kgen.return %none : !kgen.none
    }
    lit.fn @__del__[mut E](%self: !lit.ref<!String, mut E> owned_in_mem) -> !kgen.none {
      %none = kgen.param.constant: none = <#kgen.none>
      kgen.return %none : !kgen.none
    }
}

// -----

//===----------------------------------------------------------------------===//
// Struct alignment lowering
//===----------------------------------------------------------------------===//

!MyInt = !lit.struct<@struct_alignment::@MyInt>
lit.file_module @struct_alignment {
  lit.struct.decl @MyInt {
    lit.struct.field value : index
  }

  lit.fn @calculate_alignment(%my_int: !MyInt) -> index {
    %0 = kgen.param.constant: index = <64>
    kgen.return %0 : index
  }

  // Test concrete alignment on struct propagates to stack_allocation and does not
  // result in a flattened struct.
  lit.struct.decl @AlignedStruct64<my_int: !MyInt> attributes {
      minAlignment = #kgen.param.expr<apply,
                                      #kgen.symbol.constant<@struct_alignment::@calculate_alignment> : !lit.generator<("my_int": !MyInt) -> index>,
                                      #kgen.param.decl.ref<"my_int">: !MyInt> : index
  } {
    lit.struct.field value : index
  }

  // CHECK-LABEL: kgen.generator @"struct_alignment::varDeclAligned64"
  lit.fn @varDeclAligned64() {
    // CHECK-NEXT: pop.stack_allocation 1 x struct<(index) memoryOnly align(apply(:(!kgen.struct<(index) memoryOnly>) -> index @"struct_alignment::calculate_alignment", { 32 }))>
    // CHECK-SAME: align apply(:(!kgen.struct<(index) memoryOnly>) -> index @"struct_alignment::calculate_alignment", { 32 }) marked
    %a = lit.var.decl "a" var : !lit.ref<!lit.struct<@struct_alignment::@AlignedStruct64<:!MyInt {value = 32}>>, mut *"life">
    kgen.return
  }
}

// -----

// Test parametric alignment on struct propagates to stack_allocation.
!Pair = !lit.struct<@Pair>
lit.struct.decl @Pair register_passable {
  lit.struct.field first : index
  lit.struct.field second : index
}

lit.struct.decl @AlignedParam<N: !Pair> attributes {minAlignment = #lit.struct.extract<:!Pair N, "first"> : index} {
  lit.struct.field data : index
}

// CHECK-LABEL: kgen.generator @varDeclAlignedParam
lit.fn @varDeclAlignedParam<pair: !Pair>() {
  // CHECK-NEXT: pop.stack_allocation 1 x struct<(index) memoryOnly align(#kgen.struct.extract<:struct<(index, index)> pair, 0>)>
  // CHECK-SAME: align #kgen.struct.extract<:struct<(index, index)> pair, 0> marked
  %a = lit.var.decl "a" var : !lit.ref<!lit.struct<@AlignedParam<:!Pair pair>>, mut *"life">
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Struct Extensions
//===----------------------------------------------------------------------===//

// Test struct then extension (normal order)
lit.struct.decl @TestStruct1 {
}

// CHECK-LABEL: kgen.struct.generator @TestStruct1

lit.extension.decl @"extension:TestStruct1" attributes {targetStruct = @TestStruct1} {
  // CHECK-LABEL: kgen.generator @"extension:TestStruct1::extension_method"
  lit.fn @extension_method[mut O](%self: !lit.ref<!lit.struct<@TestStruct1>, mut O> read_mem) -> index {
    %result = kgen.param.constant: index = <42>
    kgen.return %result : index
  }
}

// -----

// Test extension declared before its target struct
// CHECK-LABEL: kgen.generator @"extension:TestStruct2::extension_method"

// CHECK-LABEL: kgen.struct.generator @TestStruct2
lit.extension.decl @"extension:TestStruct2" attributes {targetStruct = @TestStruct2} {
  lit.fn @extension_method[mut O](%self: !lit.ref<!lit.struct<@TestStruct2>, mut O> read_mem) -> index {
    %result = kgen.param.constant: index = <1>
    kgen.return %result : index
  }
}

lit.struct.decl @TestStruct2 {
}

// TODO(MOCO-522): Add tests for aliases in extensions into LowerLIT

// -----

// COM: Origins are pruned correctly from lit closure init ops.
!Walks = !lit.trait<@Walks>
#type_value = #kgen.type<array<1, i1>> : !kgen.type
lit.trait.decl @Walks<?, SELF: !Walks>(!Walks) {}

// CHECK-LABEL: kgen.generator @aThing
lit.fn @aThing() -> !kgen.none {
  // CHECK: kgen.closure.init()<T: type>(%arg0: !kgen.pointer<T> read_mem) -> !kgen.none
  %0 = lit.closure.init[#type_value]()<T: !Walks>[imm O1](%arg0[a]: !lit.ref<:!Walks T, imm O1> read_mem) -> !kgen.none {
    %none_0 = kgen.param.constant: none = <#kgen.none>
    kgen.return %none_0 : !kgen.none
  } : (), !lit.ref<!kgen.closure<@aThing, "aClosure" nonescaping>, mut *"aClosure`1">
  %none_1 = kgen.param.constant: none = <#kgen.none>
  kgen.return %none_1 : !kgen.none
}

// -----

// COM: LLVMMetadataArray on lit.closure.init transfers to kgen.closure.init.

#type_value = #kgen.type<array<1, i1>> : !kgen.type

// CHECK-LABEL: kgen.generator @metadata_closure
lit.fn @metadata_closure(%x: index) -> !kgen.none {
  // CHECK: LLVMArgMetadataArray = {{\[}}["nvvm.grid_constant", unit]]
  // CHECK-SAME: LLVMMetadataArray = ["nvvm.maxntid", #pop.array<256> : !pop.array<1, i32>]
  %0 = lit.closure.init[#type_value](%x)() -> index {
    kgen.return %x : index
  } : (index), !lit.ref<!kgen.closure<@metadata_closure, "fn" nonescaping>, mut *"fn`1"> {LLVMMetadataArray = ["nvvm.maxntid", #pop.array<256> : !pop.array<1, i32>], LLVMArgMetadataArray = [["nvvm.grid_constant", unit]]}
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
