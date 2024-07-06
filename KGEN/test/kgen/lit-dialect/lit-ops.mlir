// RUN: kgen-opt -split-input-file -allow-unregistered-dialect %s | kgen-opt -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: lit.struct.decl @FooStruct
// CHECK-SAME: <size, dtype: dtype, ty: type> {
// CHECK-NEXT: a : index
// CHECK-NEXT: b : !pop.scalar<dtype>
// CHECK-NEXT: c : !kgen.paramref<ty>
lit.struct.decl @FooStruct<size, dtype: dtype, ty: type> {
  lit.struct.field a : index
  lit.struct.field b : !pop.scalar<dtype>
  lit.struct.field c : !kgen.paramref<ty>
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

// CHECK-LABEL: @struct_create
// CHECK-SAME: %[[A:.*]]: index
// CHECK-SAME: %[[B:.*]]: !pop.scalar
// CHECK-SAME: %[[C:.*]]: !pop.simd
kgen.generator @struct_create<u, v: dtype>(%a: index, %b: !pop.scalar<v>, %c: !pop.simd<u, v>)
    -> !lit.struct<@FooStruct<u, :dtype v, :type !pop.simd<u, v>>> {
  // CHECK: lit.struct.create(a=%[[A]], b=%[[B]], c=%[[C]]) :
  // CHECK-SAME: (index, !pop.scalar<v>, !pop.simd<u, v>) ->
  // CHECK-SAME: !lit.struct<@FooStruct<u, :dtype v, :type simd<u, v>>>
  %0 = lit.struct.create(a=%a, b=%b, c=%c) : (index, !pop.scalar<v>, !pop.simd<u, v>) ->
    !lit.struct<@FooStruct<u, :dtype v, :type !pop.simd<u, v>>>
  kgen.return %0 : !lit.struct<@FooStruct<u, :dtype v, :type !pop.simd<u, v>>>
}

// CHECK-LABEL: @empty_struct_create
kgen.generator @empty_struct_create() -> !lit.struct<@EmptyStruct> {
  // CHECK: lit.struct.create()
  %0 = lit.struct.create() : () -> !lit.struct<@EmptyStruct>
  kgen.return %0 : !lit.struct<@EmptyStruct>
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

// CHECK-LABEL: lit.func @calls[imm a, mut b]
lit.func @calls[imm a, mut b](%arg0: !lit.signature<[2]() -> ()>) {
  // CHECK: lit.call @calls[imm a, mut b]() : !lit.signature<[2]() -> ()>
  lit.call @calls[imm a, mut b]() : !lit.signature<[2]() -> ()>
  // CHECK: lit.call_indirect %arg0[imm a, mut b]() : !lit.signature<[2]() -> ()>
  lit.call_indirect %arg0[imm a, mut b]() : !lit.signature<[2]() -> ()>
  kgen.return
}

// One implementation of dynamic_thing
// CHECK-LABEL: lit.func @vardecl
lit.func @vardecl<ty : dtype>(%x : i32) {
  // CHECK-NEXT: %a = lit.var.decl "a" imp : !lit.ref<scalar<ty>, mut life>
  %a = lit.var.decl "a" imp : !lit.ref<scalar<ty>, mut life>

  // CHECK-NEXT: %lifetime = lit.var.decl "lifetime" var : !lit.ref<index, mut lt>
  %lifetime = lit.var.decl "lifetime" var : !lit.ref<index, mut lt>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @SomeStruct<ty: dtype, n: scalar<si32> = 7>
lit.struct.decl @SomeStruct<ty: dtype, n: scalar<si32> = 7> {
  // CHECK-NEXT: lit.func @foo() {
  lit.func @foo() {
    kgen.return
  }

  // CHECK: %size = lit.var.decl "size" var : !lit.ref<scalar<ty>, mut life>
  %size = lit.var.decl "size" var : !lit.ref<scalar<ty>, mut life>

  // CHECK: lit.func @getMyType
  // CHECK-NEXT: kgen.param.constant: dtype = <ty>
  lit.func @getMyType() -> !kgen.dtype {
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
  // CHECK: lit.func @f{{.*}}
  // CHECK-NEXT:  lit.trait_func
  lit.func @f() -> !kgen.none {
    lit.trait_func
  }
}

// CHECK-LABEL: @attributesAndDecorators
lit.func @attributesAndDecorators()
  // CHECK-NEXT: decorators <{{.*}}> attributes {isParametric} {
  decorators <:() -> () @decorator> attributes {isParametric} {
  lit.end_func
}

lit.func @ref_immut<life: lifetime<1>>(%ref1: !lit.ref<@MyStruct, mut life>)
 -> !lit.ref<@MyStruct, muttoimm life> {
  // CHECK: %0 = lit.ref.immut %ref1 : <@MyStruct, mut life>
  %ref2 = lit.ref.immut %ref1: <@MyStruct, mut life>
  // CHECK: kgen.return %0 : !lit.ref<@MyStruct, muttoimm life>
  kgen.return %ref2: !lit.ref<@MyStruct, muttoimm life>
}

lit.func @ref_pointer<life: lifetime<1>, ilife: lifetime<0>>
     (%ref1: !lit.ref<@MyStruct, mut life>) {
  // CHECK: %0 = lit.ref.to_pointer %ref1 : <@MyStruct, mut life>
  %ptr = lit.ref.to_pointer %ref1: <@MyStruct, mut life>
  // CHECK: %1 = lit.ref.from_pointer %0 : <@MyStruct, imm ilife>
  %ref2 = lit.ref.from_pointer %ptr: !lit.ref<@MyStruct, imm ilife>

  // CHECK: %2 = lit.ref.to_pointer %1 : <@MyStruct, imm ilife>
  %ptr2 = lit.ref.to_pointer %ref2: !lit.ref<@MyStruct, imm ilife>
  lit.end_func
}

// CHECK-LABEL: lit.func @nested_function_region
lit.func @nested_function_region() {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: lit.func nested_fn()
    lit.func nested_fn() {
      kgen.return
    }
    hlcf.continue
  }
  kgen.return
}

// -----

// CHECK-LABEL: lit.struct.decl @A
lit.struct.decl @A {
  // CHECK-NEXT: lit.func @foo
  lit.func @foo(%self: !lit.struct<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: lit.struct.decl @B
lit.struct.decl @B {
  // CHECK-NEXT: lit.func @foo
  lit.func @foo(%self: !lit.struct<@B>, %a: !lit.struct<@A>) {
    // CHECK-NEXT: call_param[(!lit.struct<@A>) -> (): @A::@foo]
    kgen.call_param[(!lit.struct<@A>) -> (): @A::@foo](%a)

    kgen.call @A::@foo(%a) : (!lit.struct<@A>) -> ()
    kgen.return
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !lit.struct<@A>, %b: !lit.struct<@B>) {
  // CHECK-NEXT: call_param[(!lit.struct<@B>, !lit.struct<@A>) -> (): @B::@foo]
  kgen.call_param[(!lit.struct<@B>, !lit.struct<@A>) -> (): @B::@foo](%b, %a)
  // CHECK-NEXT: constant: (!lit.struct<@A>) -> () = <@A::@foo>
  %0 = kgen.param.constant: (!lit.struct<@A>) -> () = <@A::@foo>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @CrazyParams<*"m`": lifetime<0>> {
lit.struct.decl @CrazyParams<*"m`": lifetime<0>> {
}

lit.struct.decl @LifetimeRef<b: lifetime<0>> {
  lit.struct.field b : !lit.signature<(!lit.ref<@A, imm *(0,1)>) -> ()>
}

// -----

// CHECK-LABEL: lit.struct.decl @A<N>
lit.struct.decl @A<N> {
  // CHECK-NEXT: lit.func @foo<M>
  lit.func @foo<M>(%self: !lit.struct<@A<N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !lit.struct<@A<1>>) {
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
lit.func @raises_error(%raise: i1, %err: !lit.struct<@Error>, %value: !lit.struct<@Int>) -> !kgen.variant<@Error, @Int> {
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
lit.func @try_op(%err: !lit.struct<@Error>, %int: !lit.struct<@Int>) -> !lit.struct<@Int> {
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
lit.func @try_in_loop(%cond: i1) {
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
    // CHECK-NEXT: lit.func @foo(%{{.*}}: !B, %{{.*}}: !kgen.pointer<!A>
    lit.func @foo(%self: !lit.struct<@module::@B>, %a: !kgen.pointer<@module::@A>) {
      kgen.return
    }
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !kgen.pointer<@module::@A>, %b: !lit.struct<@module::@B>) {
  // CHECK-NEXT: call_param[(!B, !kgen.pointer<!A>) -> (): @module::@B::@foo]
  kgen.call_param[(!lit.struct<@module::@B>, !kgen.pointer<@module::@A>) -> (): @module::@B::@foo](%b, %a)
  kgen.return
}

lit.struct.decl @Error {}

// CHECK-LABEL: @lexical_terminators
lit.func @lexical_terminators(%cond: i1) throws -> !kgen.variant<i32, i64> {
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
  // CHECK: lit.end_func
  lit.end_func
}

// CHECK-LABEL: lit.func @async_fn() async
lit.func @async_fn() async {
  lit.end_func
}

lit.func @async_fn_byref_result(%res: !lit.ref<index, mut #lit.lifetime> byref_result) async {
  lit.end_func
}

lit.func @async_fn_throws(%err: !lit.ref<index, mut #lit.lifetime> byref_error, %res: !lit.ref<index, mut #lit.lifetime> byref_result) async|throws {
  lit.end_func
}

// CHECK-LABEL: lit.func @call_async_fn
lit.func @call_async_fn() {
  // CHECK-NEXT: lit.async.call[!lit.signature<() async -> ()>: @async_fn]()
  lit.async.call[!lit.signature<() async -> ()>: @async_fn]()
  // CHECK-NEXT: lit.async.call[!lit.signature<("res": !lit.ref<index, mut #lit.lifetime> byref_result) async -> ()>: @async_fn_byref_result]()
  lit.async.call[!lit.signature<("res": !lit.ref<index, mut #lit.lifetime> byref_result) async -> ()>: @async_fn_byref_result]()
  // CHECK-NEXT: lit.async.call[!lit.signature<("err": !lit.ref<index, mut #lit.lifetime> byref_error, "res": !lit.ref<index, mut #lit.lifetime> byref_result) throws|async -> ()>: @async_fn_throws]()
  lit.async.call[!lit.signature<("err": !lit.ref<index, mut #lit.lifetime> byref_error, "res": !lit.ref<index, mut #lit.lifetime> byref_result) async|throws -> ()>: @async_fn_throws]()
  lit.end_func
}

lit.struct.decl @GiveMeDefault {
  lit.struct.field size : !kgen.pointer<scalar<index>>
}

// CHECK-LABEL: lit.func @default_struct
// CHECK-SAME: !lit.struct<@GiveMeDefault> = {1}
lit.func @default_struct(%arg0: !lit.struct<@GiveMeDefault> = {1}) {
  kgen.return
}


lit.struct.decl @OuterParams<ty: type, fn: () -> !kgen.paramref<ty>> {
  lit.func @some_func() {
    kgen.return
  }
}

// CHECK-LABEL: lit.func @ref_it
lit.func @ref_it() {
  // CHECK: F: <type, () -> !kgen.paramref<*(1,0)>>() -> () = <@OuterParams::@some_func>
  kgen.param.declare F: <type, () -> !kgen.paramref<*(1,0)>>() -> () = <@OuterParams::@some_func>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @FuncParamStruct
// CHECK-SAME: <c: !lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()>>
lit.struct.decl @FuncParamStruct<c: !lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()>>  {
  // CHECK: lit.func @foo(%x: !kgen.pointer<@FuncParamStruct<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>>)
  lit.func @foo(%x: !kgen.pointer<@FuncParamStruct<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>>) {
    lit.end_func
  }
  // CHECK-LABEL: lit.func @bar
  lit.func @bar(%x: !kgen.pointer<@FuncParamStruct<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>>) {
    // CHECK: call @FuncParamStruct::@foo<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>(%x)
    kgen.call @FuncParamStruct::@foo<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>(%x)
    // CHECK-SAME: ("x": !kgen.pointer<@FuncParamStruct<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>>) -> ()
      : !lit.signature<("x": !kgen.pointer<@FuncParamStruct<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>>) -> ()>
    lit.end_func
  }
}

// -----

lit.struct.decl @Error {}

// CHECK-LABEL: lit.func @throwing_func
lit.func @throwing_func() throws -> i1 {
  %0 = kgen.param.constant: i1 = <0>
  // CHECK: lit.error_return %0 : i1
  lit.error_return %0 : i1
}

// CHECK: lit.globalvar.decl @global_var : !lit.struct<@Error> {
lit.globalvar.decl @global_var : !lit.struct<@Error> {
  // CHECK-NEXT: lit.globalvar.ref @global_var : <@Error, mut #lit.lifetime>
  %0 = lit.globalvar.ref @global_var : <@Error, mut #lit.lifetime>
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
// CHECK-SAME: (trait<@Trait1>, trait<@Trait2>[trait<@Trait3>])
lit.struct.decl @StructHasTraits(trait<@Trait1>, trait<@Trait2>[trait<@Trait3>]) {}

// CHECK-LABEL: lit.func @lit_loop
lit.func @lit_loop() {
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

lit.func @load_consume(%arg0 : !lit.ref<index, mut #lit.lifetime>) -> index {
  %0 = lit.load.consume %arg0 : !lit.ref<index, mut #lit.lifetime>
  kgen.return %0 : index
}
