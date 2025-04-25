// RUN: kgen-opt -split-input-file -allow-unregistered-dialect %s | kgen-opt -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: lit.struct.decl @FooStruct
// CHECK-SAME: <size, dtype: dtype, ty: type> {
// CHECK-NEXT: a : index
// CHECK-NEXT: b : !pop.scalar<dtype>
// CHECK-NEXT: c : !kgen.param<ty>
lit.struct.decl @FooStruct<size, dtype: dtype, ty: type> {
  lit.struct.field a : index
  lit.struct.field b : !pop.scalar<dtype>
  lit.struct.field c : !kgen.param<ty>
}

// CHECK-LABEL: lit.struct.decl @EmptyStruct
// CHECK-NEXT: }
lit.struct.decl @EmptyStruct {
}

// CHECK-LABEL: lit.struct.decl @ValueType
lit.struct.decl @ValueType
 // CHECK-NEXT: destructor :() -> () @ValueType::@__del__
 destructor :() -> () @ValueType::@__del__
 // CHECK-NEXT: move :() -> () @ValueType::@__moveinit__
 move :() -> () @ValueType::@__moveinit__
 // CHECK-NEXT: copy :() -> () @ValueType::@__copyinit__
 copy :() -> () @ValueType::@__copyinit__ {
}

// CHECK-LABEL: @struct_insert
kgen.generator @struct_insert(%a: index, %struct: !lit.struct<@FooStruct<2, :dtype f32, :type i32>>) {
  // CHECK: lit.struct.insert %{{.*}}, %{{.*}}[a] : index into !lit.struct<@FooStruct
  %0 = lit.struct.insert %a, %struct[a] : index into !lit.struct<@FooStruct<2, :dtype f32, :type i32>>
  kgen.return
}

// CHECK-LABEL: @struct_extract
kgen.generator @struct_extract(%struct: !lit.struct<@FooStruct<2, :dtype f32, :type i32>>) {
  // CHECK: lit.struct.extract %{{.*}}[a] : index from !lit.struct<@FooStruct
  %0 = lit.struct.extract %struct[a] : index from !lit.struct<@FooStruct<2, :dtype f32, :type i32>>
  kgen.return
}

// CHECK-LABEL: lit.fn @calls[imm a, mut b]
lit.fn @calls[imm a, mut b](%arg0: !lit.generator<[2]() -> ()>) {
  // CHECK: lit.call @calls[imm a, mut b]() : !lit.generator<[2]() -> ()>
  lit.call @calls[imm a, mut b]() : !lit.generator<[2]() -> ()>
  // CHECK: lit.call_indirect %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>
  lit.call_indirect %arg0[imm a, mut b]() : !lit.generator<[2]() -> ()>
  kgen.return
}

// One implementation of dynamic_thing
// CHECK-LABEL: lit.fn @vardecl
lit.fn @vardecl<ty : dtype>(%x : i32) {
  // CHECK-NEXT: %a = lit.var.decl "a" imp : !lit.ref<scalar<ty>, mut life>
  %a = lit.var.decl "a" imp : !lit.ref<scalar<ty>, mut life>

  // CHECK-NEXT: %origin = lit.var.decl "origin" var : !lit.ref<index, mut lt>
  %origin = lit.var.decl "origin" var : !lit.ref<index, mut lt>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @SomeStruct<ty: dtype, n: scalar<si32> = 7>
lit.struct.decl @SomeStruct<ty: dtype, n: scalar<si32> = 7> {
  // CHECK-NEXT: lit.fn @foo() {
  lit.fn @foo() {
    kgen.return
  }

  // CHECK: %size = lit.var.decl "size" var : !lit.ref<scalar<ty>, mut life>
  %size = lit.var.decl "size" var : !lit.ref<scalar<ty>, mut life>

  // CHECK: lit.fn @getMyType
  // CHECK-NEXT: kgen.param.constant: dtype = <ty>
  lit.fn @getMyType() -> !kgen.dtype {
    %dtype = kgen.param.constant: dtype = <ty>
    kgen.return %dtype : !kgen.dtype
  }
}

// CHECK-LABEL: lit.struct.decl @struct_param_passing_kinds<
// CHECK-SAME: z: dtype, |,
// CHECK-SAME: a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
// CHECK-SAME: d: dtype, e: dtype = f16, f: scalar<si16> = 2
lit.struct.decl @struct_param_passing_kinds<
  z: dtype, |,
  a: dtype, b: dtype = f32, c: scalar<si32> = 1, *,
  d: dtype, e: dtype = f16, f: scalar<si16> = 2
> {}

// CHECK-LABEL: lit.trait.decl @T {
lit.trait.decl @T {
  // CHECK: lit.fn @f{{.*}}
  // CHECK-NEXT:  kgen.unreachable
  lit.fn @f() -> !kgen.none {
    kgen.unreachable
  }
}

// CHECK-LABEL: lit.trait.decl @RP register_passable
lit.trait.decl @RP register_passable {
}

// CHECK-LABEL: @attributesAndDecorators
lit.fn @attributesAndDecorators()
  // CHECK-NEXT: decorators <{{.*}}> attributes {isParametric} {
  decorators <:() -> () @decorator> attributes {isParametric} {
  lit.end_fn
}

// CHECK-LABEL: @end_fn
lit.fn @end_fn() {
  // CHECK-NEXT: lit.end_fn unresolved
  lit.end_fn unresolved
}


lit.fn @ref_immut<life: origin<1>>(%ref1: !lit.ref<@MyStruct, mut life>)
 -> !lit.ref<@MyStruct, muttoimm life> {
  // CHECK: %0 = lit.ref.immut %ref1 : <@MyStruct, mut life>
  %ref2 = lit.ref.immut %ref1: <@MyStruct, mut life>
  // CHECK: kgen.return %0 : !lit.ref<@MyStruct, muttoimm life>
  kgen.return %ref2: !lit.ref<@MyStruct, muttoimm life>
}

lit.fn @ref_pointer<life: origin<1>, ilife: origin<0>>
     (%ref1: !lit.ref<@MyStruct, mut life>) {
  // CHECK: %0 = lit.ref.to_pointer %ref1 : <@MyStruct, mut life>
  %ptr = lit.ref.to_pointer %ref1: <@MyStruct, mut life>
  // CHECK: %1 = lit.ref.from_pointer %0 : <@MyStruct, imm ilife>
  %ref2 = lit.ref.from_pointer %ptr: !lit.ref<@MyStruct, imm ilife>

  // CHECK: %2 = lit.ref.to_pointer %1 : <@MyStruct, imm ilife>
  %ptr2 = lit.ref.to_pointer %ref2: !lit.ref<@MyStruct, imm ilife>
  lit.end_fn
}

// CHECK-LABEL: lit.fn @nested_function_region
lit.fn @nested_function_region() {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: lit.fn nested_fn()
    lit.fn nested_fn() {
      kgen.return
    }
    hlcf.continue
  }
  kgen.return
}

// -----

// CHECK-LABEL: lit.struct.decl @A
lit.struct.decl @A {
  // CHECK-NEXT: lit.fn @foo
  lit.fn @foo(%self: !lit.struct<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: lit.struct.decl @B
lit.struct.decl @B {
  // CHECK-NEXT: lit.fn @foo
  lit.fn @foo(%self: !lit.struct<@B>, %a: !lit.struct<@A>) {
    // CHECK-NEXT: call_param[(!lit.struct<@A>) -> (): @A::@foo]
    kgen.call_param[(!lit.struct<@A>) -> (): @A::@foo](%a)

    kgen.call @A::@foo(%a) : (!lit.struct<@A>) -> ()
    kgen.return
  }
}

// CHECK-LABEL: lit.fn @main
lit.fn @main(%a: !lit.struct<@A>, %b: !lit.struct<@B>) {
  // CHECK-NEXT: call_param[(!lit.struct<@B>, !lit.struct<@A>) -> (): @B::@foo]
  kgen.call_param[(!lit.struct<@B>, !lit.struct<@A>) -> (): @B::@foo](%b, %a)
  // CHECK-NEXT: constant: (!lit.struct<@A>) -> () = <@A::@foo>
  %0 = kgen.param.constant: (!lit.struct<@A>) -> () = <@A::@foo>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @CrazyParams<*"m`": origin<0>> {
lit.struct.decl @CrazyParams<*"m`": origin<0>> {
}

lit.struct.decl @LifetimeRef<b: origin<0>> {
  lit.struct.field b : !lit.generator<(!lit.ref<@A, imm *(0,1)>) -> ()>
}

// -----

// CHECK-LABEL: lit.struct.decl @A<N>
lit.struct.decl @A<N> {
  // CHECK-NEXT: lit.fn @foo<M>
  lit.fn @foo<M>(%self: !lit.struct<@A<N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: lit.fn @main
lit.fn @main(%a: !lit.struct<@A<1>>) {
  // CHECK-NEXT: call_param[(!lit.struct<@A<1>>) -> index: @A::@foo<1, 2>]
  %- = kgen.call_param[(!lit.struct<@A<1>>) -> index: @A::@foo<1, 2>](%a)
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @NoFields {
// CHECK-NEXT: }
lit.struct.decl @NoFields {}

// COM: Types from the standard library.
lit.struct.decl @Error {}
lit.struct.decl @Int {}

// CHECK-LABEL: @raises_error
lit.fn @raises_error(%raise: i1, %err: !lit.struct<@Error>, %value: !lit.struct<@Int>) -> !kgen.variant<@Error, @Int> {
  hlcf.if %raise {
    // CHECK: %[[ERR:.*]] = kgen.variant.create %err
    %result = kgen.variant.create %err, 0 : <@Error, @Int>
    // CHECK: kgen.return %[[ERR]]
    kgen.return %result : !kgen.variant<@Error, @Int>
  } else {
    hlcf.yield
  }
  // CHECK: %[[VALUE:.*]] = kgen.variant.create %value
  %result = kgen.variant.create %value, 1 : <@Error, @Int>
  // CHECK: kgen.return %[[VALUE]]
  kgen.return %result : !kgen.variant<@Error, @Int>
}

// CHECK-LABEL: @try_op
lit.fn @try_op(%err: !lit.struct<@Error>, %int: !lit.struct<@Int>) -> !lit.struct<@Int> {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: } except (%{{.*}}: !lit.struct<@Error>) {
  } except (%exception: !lit.struct<@Error>) {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: } finally {
  } finally {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  }
  kgen.return %int : !lit.struct<@Int>
}

// CHECK-LABEL: @try_in_loop
lit.fn @try_in_loop(%cond: i1) {
  // CHECK-NEXT: lit.loop
  lit.loop cond {
    lit.loop.condition %cond : i1
  } body {
    // CHECK: lit.try
    lit.try {
      // CHECK-NEXT: hlcf.if
      hlcf.if %cond {
        // CHECK-NEXT: hlcf.break
        hlcf.break
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: hlcf.yield
        hlcf.yield
      }
      // CHECK: lit.try.yield
      lit.try.yield
    // CHECK-NEXT: except
    } except (%arg0: !lit.struct<@Error>) {
      // CHECK-NEXT: hlcf.break
      hlcf.break
    // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: lit.try.yield
      lit.try.yield
    // CHECK-NEXT: finally
    } finally {
      // CHECK-NEXT: lit.try.yield
      lit.try.yield
    }
    // CHECK: lit.loop.continue
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  // CHECK: kgen.return
  kgen.return
}

// -----

// CHECK-DAG: !B = !lit.struct<@module::@B>
// CHECK-DAG: !A = !lit.struct<@module::@A>

// CHECK-LABEL: lit.file_module @module
lit.file_module @module {
  // CHECK: lit.struct.decl @A
  lit.struct.decl @A {}

  // CHECK: lit.struct.decl @B
  lit.struct.decl @B {
    // CHECK-NEXT: lit.fn @foo(%{{.*}}: !B, %{{.*}}: !kgen.pointer<!A>
    lit.fn @foo(%self: !lit.struct<@module::@B>, %a: !kgen.pointer<@module::@A>) {
      kgen.return
    }
  }
}

// CHECK-LABEL: lit.fn @main
lit.fn @main(%a: !kgen.pointer<@module::@A>, %b: !lit.struct<@module::@B>) {
  // CHECK-NEXT: call_param[(!B, !kgen.pointer<!A>) -> (): @module::@B::@foo]
  kgen.call_param[(!lit.struct<@module::@B>, !kgen.pointer<@module::@A>) -> (): @module::@B::@foo](%b, %a)
  kgen.return
}

lit.struct.decl @Error {}

// CHECK-LABEL: @lexical_terminators
lit.fn @lexical_terminators(%cond: i1) throws -> !kgen.variant<i32, i64> {
  // CHECK: lit.loop
  lit.loop cond {
    lit.loop.condition %cond : i1
  } body {
    // CHECK: hlcf.if
    hlcf.if %cond {
      // CHECK-NEXT: lit.break
      lit.break
      hlcf.yield
    // CHECK: else
    } else {
      // CHECK-NEXT: lit.continue
      lit.continue
      hlcf.yield
    }
    lit.loop.continue
  } else {
    lit.loop.yield
  }

  // CHECK: lit.try
  lit.try {
    // CHECK: lit.raise
    lit.raise
    lit.try.yield
  } except {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  // CHECK: lit.end_fn
  lit.end_fn
}

// CHECK-LABEL: lit.fn @async_fn() async
lit.fn @async_fn() async {
  lit.end_fn
}

lit.fn @async_fn_byref_result(%res: !lit.ref<index, mut #lit.any.origin> byref_result) async {
  lit.end_fn
}

lit.fn @async_fn_throws(%err: !lit.ref<index, mut #lit.any.origin> byref_error, %res: !lit.ref<index, mut #lit.any.origin> byref_result) async|throws {
  lit.end_fn
}

// CHECK-LABEL: lit.fn @call_async_fn
lit.fn @call_async_fn() {
  // CHECK-NEXT: lit.async.call[!lit.generator<() async -> ()>: @async_fn]()
  lit.async.call[!lit.generator<() async -> ()>: @async_fn]()
  // CHECK-NEXT: lit.async.call[!lit.generator<("res": !lit.ref<index, mut #lit.any.origin> byref_result) async -> ()>: @async_fn_byref_result]()
  lit.async.call[!lit.generator<("res": !lit.ref<index, mut #lit.any.origin> byref_result) async -> ()>: @async_fn_byref_result]()
  // CHECK-NEXT: lit.async.call[!lit.generator<("err": !lit.ref<index, mut #lit.any.origin> byref_error, "res": !lit.ref<index, mut #lit.any.origin> byref_result) throws|async -> ()>: @async_fn_throws]()
  lit.async.call[!lit.generator<("err": !lit.ref<index, mut #lit.any.origin> byref_error, "res": !lit.ref<index, mut #lit.any.origin> byref_result) async|throws -> ()>: @async_fn_throws]()
  lit.end_fn
}

lit.struct.decl @GiveMeDefault {
  lit.struct.field size : !kgen.pointer<scalar<index>>
}

// CHECK-LABEL: lit.fn @default_struct
// CHECK-SAME: !lit.struct<@GiveMeDefault> = {1}
lit.fn @default_struct(%arg0: !lit.struct<@GiveMeDefault> = {1}) {
  kgen.return
}


lit.struct.decl @OuterParams<ty: type, fn: () -> !kgen.param<ty>> {
  lit.fn @some_func() {
    kgen.return
  }
}

// CHECK-LABEL: lit.fn @ref_it
lit.fn @ref_it() {
  // CHECK: F: <type, () -> !kgen.param<*(1,0)>>() -> () = <@OuterParams::@some_func>
  kgen.param.declare F: <type, () -> !kgen.param<*(1,0)>>() -> () = <@OuterParams::@some_func>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @FuncParamStruct
// CHECK-SAME: <c: !lit.generator<<type>(!kgen.param<*(0,0)>) -> ()>>
lit.struct.decl @FuncParamStruct<c: !lit.generator<<type>(!kgen.param<*(0,0)>) -> ()>>  {
  // CHECK: lit.fn @foo(%x: !kgen.pointer<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>)
  lit.fn @foo(%x: !kgen.pointer<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>) {
    lit.end_fn
  }
  // CHECK-LABEL: lit.fn @bar
  lit.fn @bar(%x: !kgen.pointer<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>) {
    // CHECK: call @FuncParamStruct::@foo<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>(%x)
    kgen.call @FuncParamStruct::@foo<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>(%x)
    // CHECK-SAME: ("x": !kgen.pointer<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>) -> ()
      : !lit.generator<("x": !kgen.pointer<@FuncParamStruct<:!lit.generator<<type>(!kgen.param<*(0,0)>) -> ()> c>>) -> ()>
    lit.end_fn
  }
}

// -----

lit.struct.decl @Error {}

// CHECK-LABEL: lit.fn @throwing_func
lit.fn @throwing_func() throws -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  // CHECK: lit.error_return %0 : i1
  lit.error_return %0 : i1
}

// CHECK: lit.globalvar.decl @global_var : !lit.struct<@Error> {
lit.globalvar.decl @global_var : !lit.struct<@Error> {
  // CHECK-NEXT: lit.globalvar.ref @global_var : <@Error, mut #lit.any.origin>
  %0 = lit.globalvar.ref @global_var : <@Error, mut #lit.any.origin>
// CHECK-NEXT: }, {
}, {
// CHECK-NEXT: }
}

// -----

#file = #debuginfo.file<"foo.mlir" in "">
#loc = loc("foo.mlir":7:8)

// CHECK-LABEL: lit.struct.decl @Foo
lit.struct.decl @Foo {
  lit.struct.field value : index
} loc(fused<#file>[#loc])

// -----

// struct with traits
// CHECK-LABEL: lit.trait.decl @Trait1
lit.trait.decl @Trait1 {}
lit.trait.decl @Trait2 {}
lit.trait.decl @Trait3 {}

// CHECK-LABEL: lit.struct.decl @StructHasTraits
// CHECK-SAME: (trait<@Trait1, @Trait2, @Trait3>)
lit.struct.decl @StructHasTraits(trait<@Trait1, @Trait2, @Trait3>) {}

// CHECK-LABEL: lit.fn @lit_loop
lit.fn @lit_loop() {
  lit.loop cond {
    %0 = index.bool.constant true
    // CHECK: lit.loop.condition %{{.*}}: i1
    lit.loop.condition %0: i1
  } body {
    // CHECK: lit.loop.continue
    lit.loop.continue
  } else {
    // CHECK: lit.loop.yield
    lit.loop.yield
  } {unrollLevel = #hlcf<unroll_level full>}

  kgen.return
}

// -----

lit.fn @load_consume(%arg0 : !lit.ref<index, mut #lit.any.origin>) -> index {
  %0 = lit.load.consume %arg0 : !lit.ref<index, mut #lit.any.origin>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: lit.fn @make_closure
  lit.fn @make_closure[imm Y, imm Z](%y: !lit.ref<@String, imm Y> owned_in_mem, %x: index, %z: !lit.ref<@String, imm Z> owned_in_mem) -> !kgen.none {
    // CHECK: lit.closure.init(%y[ref: imm Y], %x, %z[@String::@__copyinit__[2](!lit.ref<@String, mut *[0,0]>, !lit.ref<@String, imm *[0,1]>)])(%arg0[y2]: index) -> index
    // CHECK: } : (!lit.ref<@String, imm Y>, index, !lit.ref<@String, imm Z>), !lit.ref<!kgen.closure<@make_closure, "foo" nonescaping>, mut C>
    %0 = lit.closure.init(%y[ref: imm Y], %x, %z[@String::@__copyinit__[2](!lit.ref<@String, mut *[0,0]>, !lit.ref<@String, imm *[0,1]>)])(%arg0[y2]: index) -> index {
      lit.end_fn
    } : (!lit.ref<@String, imm Y>, index, !lit.ref<@String, imm Z>),
        !lit.ref<!kgen.closure<@make_closure, "foo" nonescaping>, mut C>
    // CHECK: lit.closure.init()(%arg0[y2]: index) -> index
    // CHECK: } : (), !lit.ref<!kgen.closure<@make_closure, "foo" registerpassable>, mut C2>
    %1 = lit.closure.init()(%arg0[y2]: index) -> index {
      lit.end_fn
    } : (), !lit.ref<!kgen.closure<@make_closure, "foo" registerpassable>, mut C2>
    lit.end_fn
  }

  lit.struct.decl @String {

  }

// -----

// COM: Ensure Closure Symbols Are Valid VTable Entries

!Closure = !lit.trait<@Closure>
!String = !lit.struct<@String>
!Impl = !kgen.closure<@make_closure, "foo" nonescaping>

// CHECK: #type_value = #kgen.type<!kgen.closure<@make_closure, "foo" nonescaping>, {"__call__" : !lit.generator<[1]("self": !lit.ref<!kgen.closure<@make_closure, "foo" nonescaping>, imm *[0,0]> read_mem, "y": index) -> index> = #kgen.closure.symbol<@make_closure, "foo", #kgen.closure_method<call>>}> : !lit.trait<@Closure>
#Impl1 = #kgen.type<!Impl, {"__call__" :
                            !lit.generator<[1]("self": !lit.ref<!Impl, imm *[0,0]> read_mem, "y": index) -> index> =
                            #kgen.closure.symbol<@make_closure, "foo", #kgen.closure_method<call>>}> : !Closure

lit.trait.decl @Closure<?, SELF: !Closure> {
  lit.fn @"__call__"[imm O](%self: !lit.ref<:!Closure SELF, imm O> read_mem, %y: index) -> index {
    kgen.unreachable
  }
}

lit.fn @make_closure[imm Y, imm Z](%y: !lit.ref<!String, imm Y> owned_in_mem, %x:index, %z: !lit.ref<!String, imm Z> owned_in_mem) -> !kgen.none {
  %impl = lit.closure.init(%y[ref: imm Y], %x, %z[@String::@__copyinit__[2](!lit.ref<!String, imm *[0,0]>, !lit.ref<!String, imm *[0,1]>)])(%y2: index) -> index {
    lit.end_fn
  } : (!lit.ref<!String, imm Y>, index, !lit.ref<!String, imm Z>), !lit.ref<!kgen.closure<@make_closure, "foo" nonescaping>, mut C>
  %2 = lit.call @direct[mut C]<:!Closure #Impl1>(%impl, %x) :
           !lit.generator<[1]("c":!lit.ref<:!Closure #Impl1, mut *[0,0]> read_mem, "x": index) -> !kgen.none>
  lit.end_fn
}

lit.fn @direct<CT: !Closure>[mut Origin0](%c: !lit.ref<:!Closure CT, mut Origin0> read_mem, %x: index) -> !kgen.none {
   %0 = lit.call[!lit.generator<[1]("self": !lit.ref<:!Closure CT, imm *[0,0]> read_mem, "y": index) -> index>:
        get_vtable_entry(:!Closure CT, "__call__")][mut Origin0](%c, %x)
   lit.end_fn
}

lit.struct.decl @String
    copy :!lit.generator<[2]("existing": !lit.ref<!String, imm *[0,0]> read_mem, ?, "self": !lit.ref<!String, mut *[0,1]> byref_result) -> !kgen.none> @String::@"__copyinit__" {
    lit.fn @__copyinit__[imm E1, mut E2](%existing: !lit.ref<!String, imm E2> read_mem, ?, %self: !lit.ref<!String, mut E1> byref_result) -> !kgen.none {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_fn
    }
}

// -----

// COM: Ensure lifetimes and parameters and parsed/printed correctly.
// Note that symbols of copy/move captures are specified similar to a lit.call:
// (1) lifetimes
// (2) the values bound to the parameters of the symbol
// (3) the existing and self types
// The existing and self types are provided to avoid parameter inference from the capture types.

lit.struct.decl @Foo<PARAM: index>
  copy :!lit.generator<<"P": index, "Q": index>[2]("existing": !lit.ref<@Foo<:index *(0,0)>, imm *[0,0]> read_mem, ?, "self": !lit.ref<@Foo<:index *(0,0)>, mut *[0,1]> byref_result) -> !kgen.none> @Foo::@__copyinit__ {
  lit.struct.field x : index
  lit.struct.field y : index
  lit.fn @__copyinit__<P: index, Q: index>[imm O1, mut O2](%existing: !lit.ref<@Foo<:index P>, imm O1> read_mem, ?, %self: !lit.ref<@Foo<:index P>, mut O2> byref_result) -> !kgen.none {
    lit.end_fn
  }
}

lit.fn @"bar"<PARAM: index>[mut R](?, %__result__: !lit.ref<@Foo<:index PARAM>, mut R> byref_result) -> !kgen.none {
  %foo = lit.var.decl "foo" var : !lit.ref<@Foo<:index PARAM>, mut FOO>
  // CHECK: lit.closure.init(%foo[@Foo::@__copyinit__[2]<PARAM, 2>(!lit.ref<@Foo<*(0,0)>, imm *[0,0]>, !lit.ref<@Foo<*(0,0)>, mut *[0,1]>)])(%arg0[y2]: index) -> index {
  // CHECK-NEXT: lit.end_fn
  // CHECK-NEXT: } : (!lit.ref<@Foo<PARAM>, mut FOO>), !lit.ref<!kgen.closure<@make_closure, "foo" nonescaping>, mut C>
  %impl = lit.closure.init(%foo[@Foo::@__copyinit__[2]<PARAM, 2>(!lit.ref<@Foo<:index *(0,0)>, imm *[0,0]>, !lit.ref<@Foo<:index *(0,0)>, mut *[0,1]>)])(%y2: index) -> index {
    lit.end_fn
  } : (!lit.ref<@Foo<:index PARAM>, mut FOO>), !lit.ref<!kgen.closure<@make_closure, "foo" nonescaping>, mut C>
  lit.end_fn
}
