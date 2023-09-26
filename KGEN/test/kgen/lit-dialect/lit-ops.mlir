// RUN: kgen-opt -split-input-file -allow-unregistered-dialect %s | kgen-opt -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: lit.func @trivial_generator(%name: si32)
lit.func @trivial_generator(%name: si32) -> si32 {
  // CHECK-NEXT: kgen.return %name : si32
  kgen.return %name : si32
}

// One implementation of dynamic_thing
// CHECK-LABEL: lit.func @vardecl
lit.func @vardecl<ty : dtype>(%x : i32) {
  // CHECK-NEXT: %a = lit.varlet.decl "a" var synth : !lit.ref<mut scalar<ty>, life>
  %a = lit.varlet.decl "a" var synth : !lit.ref<mut scalar<ty>, life>

  // CHECK-NEXT: %lifetime = lit.varlet.decl "lifetime" : !lit.ref<mut index, lt>
  %lifetime = lit.varlet.decl "lifetime" : !lit.ref<mut index, lt>

  // CHECK-NEXT: %y = lit.letreg.decl "y" = %x : i32
  %y = lit.letreg.decl "y" = %x: i32

  // CHECK-NEXT: %z = lit.letreg.decl "z" = %y : i32
  %z = lit.letreg.decl "z" = %y: i32
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @SomeStruct<ty: dtype, n: scalar<si32> = 7>
lit.struct.decl @SomeStruct<ty: dtype, n: scalar<si32> = 7> {
  // CHECK-NEXT: lit.func @foo() {
  lit.func @foo() {
    kgen.return
  }

  // CHECK: %size = lit.varlet.decl "size" var : !lit.ref<mut scalar<ty>, life>
  %size = lit.varlet.decl "size" var : !lit.ref<mut scalar<ty>, life>

  // CHECK: lit.func @getMyType
  // CHECK-NEXT: kgen.param.constant: dtype = <ty>
  lit.func @getMyType() -> !kgen.dtype {
    %dtype = kgen.param.constant: dtype = <ty>
    kgen.return %dtype : !kgen.dtype
  }
}

// CHECK-LABEL: lit.trait.decl @T {
lit.trait.decl @T {
  // CHECK: lit.func @f{{.*}}
  // CHECK-NEXT:  lit.trait_func
  lit.func @f() -> !lit.none {
    lit.trait_func
  }
}

// CHECK-LABEL: @noneTypeAndValue
lit.func @noneTypeAndValue() -> !lit.none {
  // CHECK-NEXT: kgen.param.constant: !lit.none = <#lit.none>
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  kgen.return %0 : !lit.none
}

// CHECK-LABEL: @attributesAndDecorators
lit.func @attributesAndDecorators()
  // CHECK-NEXT: decorators <{{.*}}> attributes {isParametric} {
  decorators <:() -> () @decorator> attributes {isParametric} {
  lit.end_func
}

// -----

// CHECK-LABEL: lit.struct.decl @A
lit.struct.decl @A {
  // CHECK-NEXT: lit.func @foo
  lit.func @foo(%self: !kgen.declref<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: lit.struct.decl @B
lit.struct.decl @B {
  // CHECK-NEXT: lit.func @foo
  lit.func @foo(%self: !kgen.declref<@B>, %a: !kgen.declref<@A>) {
    // CHECK-NEXT: call_param[(!kgen.declref<@A>) -> (): @A::@foo]
    kgen.call_param[(!kgen.declref<@A>) -> (): @A::@foo](%a)

    kgen.call @A::@foo(%a) : (!kgen.declref<@A>) -> ()
    kgen.return
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !kgen.declref<@A>, %b: !kgen.declref<@B>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@B>, !kgen.declref<@A>) -> (): @B::@foo]
  kgen.call_param[(!kgen.declref<@B>, !kgen.declref<@A>) -> (): @B::@foo](%b, %a)
  // CHECK-NEXT: constant: (!kgen.declref<@A>) -> () = <@A::@foo>
  %0 = kgen.param.constant: (!kgen.declref<@A>) -> () = <@A::@foo>
  kgen.return
}

// -----

// CHECK-LABEL: lit.struct.decl @A<N>
lit.struct.decl @A<N> {
  // CHECK-NEXT: lit.func @foo<M>
  lit.func @foo<M>(%self: !kgen.declref<@A<N = N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !kgen.declref<@A<N = 1>>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<1, 2>]
  %- = kgen.call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<1, 2>](%a)
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @NoFields {
// CHECK-NEXT: }
lit.struct.decl @NoFields {}

// COM: Types from the standard library.
lit.struct.decl @Error {}
lit.struct.decl @Int {}

// CHECK-LABEL: @raises_error
lit.func @raises_error(%raise: i1, %err: !kgen.declref<@Error>, %value: !kgen.declref<@Int>) -> !pop.variant<@Error, @Int> {
  hlcf.if %raise {
    // CHECK: %[[ERR:.*]] = pop.variant.create %err
    %result = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, @Int>
    // CHECK: kgen.return %[[ERR]]
    kgen.return %result : !pop.variant<@Error, @Int>
  } else {
    hlcf.yield
  }
  // CHECK: %[[VALUE:.*]] = pop.variant.create %value
  %result = pop.variant.create %value : !kgen.declref<@Int> -> !pop.variant<@Error, @Int>
  // CHECK: kgen.return %[[VALUE]]
  kgen.return %result : !pop.variant<@Error, @Int>
}

// CHECK-LABEL: @try_op
lit.func @try_op(%err: !kgen.declref<@Error>, %int: !kgen.declref<@Int>) -> !kgen.declref<@Int> {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: } except (%{{.*}}: !kgen.declref<@Error>) {
  } except (%exception: !kgen.declref<@Error>) {
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
  kgen.return %int : !kgen.declref<@Int>
}

// CHECK-LABEL: @try_in_loop
lit.func @try_in_loop(%cond: i1) {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: lit.try
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
    } except (%arg0: !kgen.declref<@Error>) {
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
    // CHECK: hlcf.continue
    hlcf.continue
  }
  // CHECK: kgen.return
  kgen.return
}

// -----

// CHECK-LABEL: lit.file_module @module
lit.file_module @module {
  // CHECK: lit.struct.decl @A
  lit.struct.decl @A {}

  // CHECK: lit.struct.decl @B
  lit.struct.decl @B {
    // CHECK-NEXT: lit.func @foo(%{{.*}}: !kgen.declref<@module::@B>, %{{.*}}: !kgen.pointer<@module::@A>
    lit.func @foo(%self: !kgen.declref<@module::@B>, %a: !kgen.pointer<@module::@A>) {
      kgen.return
    }
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !kgen.pointer<@module::@A>, %b: !kgen.declref<@module::@B>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@module::@B>, !kgen.pointer<@module::@A>) -> (): @module::@B::@foo]
  kgen.call_param[(!kgen.declref<@module::@B>, !kgen.pointer<@module::@A>) -> (): @module::@B::@foo](%b, %a)
  kgen.return
}

lit.struct.decl @Error {}

// CHECK-LABEL: @lexical_terminators
lit.func @lexical_terminators(%cond: i1, %err: !kgen.declref<@Error>) throws -> !pop.variant<i32, i64> {
  // CHECK: hlcf.loop
  hlcf.loop {
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
    hlcf.continue
  }
  // CHECK: lit.try
  lit.try {
    // CHECK-NEXT: lit.raise %err : <@Error>
    lit.raise %err : <@Error>
    lit.try.yield
  } except (%e: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  // CHECK: lit.raise %err : <@Error>
  lit.try {
    lit.raise %err : <@Error>
    lit.try.yield
  } except (%e: !kgen.declref<@Error>) {
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

// CHECK-LABEL: lit.func @call_async_fn
lit.func @call_async_fn() {
  // CHECK-NEXT: lit.async.call[() async -> (): @async_fn]()
  %0 = lit.async.call[() async -> (): @async_fn]()
  lit.end_func
}

// CHECK-LABEL: lit.func @async_execute
lit.func @async_execute() -> !pop.coroutine<() -> (i32, i64)> {
  %0 = kgen.param.constant: i32 = <3>
  // CHECK: %[[HDL:.*]] = lit.async.execute <() -> (i32, i64)> {
  %coroHdl = lit.async.execute <() -> (i32, i64)> {
    %1 = kgen.param.constant: i64 = <5>
    // CHECK: lit.async.return %0, %2 : i32, i64
    lit.async.return %0, %1 : i32, i64
  }
  // CHECK: kgen.return %[[HDL]]
  kgen.return %coroHdl : !pop.coroutine<() -> (i32, i64)>
}

// CHECK-LABEL: lit.func @param_return
lit.func @param_return<() -> r0: dtype, r1>() {
  // CHECK-NEXT lit.param_return<:dtype si32, 2>
  lit.param_return<:dtype si32, 2>
  lit.end_func
}

// CHECK-LABEL: lit.func @param_return_no_results
lit.func @param_return_no_results<() -> ()>() {
  // CHECK-NEXT: lit.param_return
  lit.param_return
  lit.end_func
}

lit.struct.decl @GiveMeDefault {
  lit.struct.field size : !kgen.pointer<scalar<index>>
}

// CHECK-LABEL: lit.func @default_struct
// CHECK-SAME: !kgen.declref<@GiveMeDefault> = #lit.struct<{value = 1}>
lit.func @default_struct(%arg0: !kgen.declref<@GiveMeDefault> = #lit.struct<{value = 1}>) {
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
  // CHECK: lit.func @foo(%x: !kgen.pointer<@FuncParamStruct<c: !lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> = c>>)
  lit.func @foo(%x: !kgen.pointer<@FuncParamStruct<c: !lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> = c>>) {
    lit.end_func
  }
  // CHECK-LABEL: lit.func @bar
  lit.func @bar(%x: !kgen.pointer<@FuncParamStruct<c: !lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> = c>>) {
    // CHECK: call @FuncParamStruct::@foo<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>(%x)
    kgen.call @FuncParamStruct::@foo<:!lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> c>(%x)
    // CHECK-SAME: ("x": !kgen.pointer<@FuncParamStruct<c: !lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> = c>>) -> ()
      : !lit.signature<("x": !kgen.pointer<@FuncParamStruct<c: !lit.signature<<type>(!kgen.paramref<*(0,0)>) -> ()> = c>>) -> ()>
    lit.end_func
  }
}

// -----

lit.func @throwing_caller() throws -> !pop.variant<@Error, !lit.none> {
  %y = lit.varlet.decl "y" var : !lit.ref<mut @MyStruct, *"life">
  %none = kgen.param.constant: !lit.none = <#lit.none>
  %ret = kgen.param.constant: !pop.variant<@Error, !lit.none> = <#pop.variant<:!lit.none #lit.none>>
  %yptr = lit.ref.to_pointer %y: !lit.ref<mut @MyStruct, *"life">
  // CHECK: lit.handle_variant %variant, %1 : (!pop.variant<@Error, !lit.none>, !kgen.pointer<@MyStruct>) -> !lit.none {
  %0 = lit.handle_variant %ret, %yptr : (!pop.variant<@Error, !lit.none>, !kgen.pointer<@MyStruct>) -> !lit.none {
    // CHECK-NEXT: lit.yield %{{.*}} : !lit.none
    lit.yield %none : !lit.none
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: return %variant
    kgen.return %ret : !pop.variant<@Error, !lit.none>
  }
  kgen.return %ret : !pop.variant<@Error, !lit.none>
}

// -----

lit.struct.decl @Error {}

lit.func @throwing_func() throws -> !pop.variant<@Error, !lit.none> {
  %1 = lit.struct.create() : () -> !kgen.declref<@Error>
  %2 = pop.variant.create %1 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
  // CHECK: lit.error_return %{{.*}} : <@Error, !lit.none>
  lit.error_return %2 : !pop.variant<@Error, !lit.none>
}

// CHECK: lit.globalvar.decl @global_var : !kgen.declref<@Error> {
lit.globalvar.decl @global_var : !kgen.declref<@Error> {
  // CHECK-NEXT: lit.globalvar.ref @global_var : <@Error>
  %0 = lit.globalvar.ref @global_var : <@Error>
// CHECK-NEXT: }, {
}, {
// CHECK-NEXT: }
}

// CHECK: lit.globalvar.decl @global_let : !kgen.declref<@Error> isVar
lit.globalvar.decl @global_let : !kgen.declref<@Error> isVar {
}, {
  %0 = lit.globalvar.ref @global_let : <@Error>
}

// -----

#file = #debuginfo.file<"foo.mlir" in "">
#loc = loc("foo.mlir":7:8)

// CHECK-LABEL: lit.struct.decl @Foo
lit.struct.decl @Foo {
  lit.struct.field value : index
} loc(fused<#file>[#loc])
