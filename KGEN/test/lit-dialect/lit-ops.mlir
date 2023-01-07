// RUN: kgen-opt -split-input-file -allow-unregistered-dialect %s | kgen-opt -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: lit.func @trivial_generator(%name: si32)
lit.func @trivial_generator(%name: si32) -> si32 {
  // CHECK-NEXT: kgen.return %name : si32
  kgen.return %name : si32
}

// CHECK-LABEL: kgen.generator.interface @itf<ty: dtype>(!pop.scalar<ty>) -> !pop.scalar<ty>
kgen.generator.interface @itf<ty : dtype>(!pop.scalar<ty>) -> !pop.scalar<ty>

// One implementation of dynamic_thing
// CHECK-LABEL: lit.func @impl1<ty: dtype>(%arg0: !pop.scalar<ty>
// CHECK-NEXT: implements @itf {
lit.func @impl1<ty : dtype>(%arg0: !pop.scalar<ty>) -> !pop.scalar<ty>
  implements @itf {
  kgen.return %arg0 : !pop.scalar<ty>
}

// One implementation of dynamic_thing
// CHECK-LABEL: lit.func @vardecl
lit.func @vardecl<ty : dtype>(%x : i32) {
// CHECK-NEXT: %a = lit.var.decl "a" : <scalar<ty>>
  %a = lit.var.decl "a": !pop.pointer<scalar<ty>>

  // CHECK-NEXT: %y = lit.let.decl "y" = %x : i32
  %y = lit.let.decl "y" = %x: i32

  // CHECK-NEXT: %z = lit.let.decl "z" = %y : i32
  %z = lit.let.decl "z" = %y: i32
  kgen.return
}

// CHECK-LABEL: kgen.struct.decl @SomeStruct<ty: dtype> {
kgen.struct.decl @SomeStruct<ty: dtype> {
  // CHECK-NEXT: lit.func @foo() {
  lit.func @foo() {
    kgen.return
  }

  // CHECK: %size = lit.var.decl "size" : <scalar<ty>>
  %size = lit.var.decl "size" : !pop.pointer<scalar<ty>>

  // CHECK: lit.func @getMyType
  // CHECK-NEXT: kgen.param.constant: dtype = <ty>
  lit.func @getMyType() -> !kgen.dtype {
    %dtype = kgen.param.constant: dtype = <ty>
    kgen.return %dtype : !kgen.dtype
  }

  // CHECK: lit.func @shadowParameter<ty>
  lit.func @shadowParameter<ty>() {
    // CHECK-NEXT: kgen.param.constant = <ty>
    %0 = kgen.param.constant = <ty>
    kgen.return
  }
}

// CHECK-LABEL: @noneTypeAndValue
lit.func @noneTypeAndValue() -> !lit.none {
  // CHECK-NEXT: kgen.param.constant: !lit.none = <#lit.none>
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  kgen.return %0 : !lit.none
}

// -----

// CHECK-LABEL: kgen.struct.decl @A
kgen.struct.decl @A {
  // CHECK-NEXT: lit.func @foo
  lit.func @foo(%self: !kgen.declref<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: kgen.struct.decl @B
kgen.struct.decl @B {
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

// CHECK-LABEL: kgen.struct.decl @A<N>
kgen.struct.decl @A<N> {
  // CHECK-NEXT: lit.func @foo<M>
  lit.func @foo<M>(%self: !kgen.declref<@A<N = N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !kgen.declref<@A<N = 1>>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<N = 1, M = 2>]
  %- = kgen.call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<N = 1, M = 2>](%a)
  kgen.return
}

// CHECK-LABEL: kgen.struct.decl @NoFields {
// CHECK-NEXT: }
kgen.struct.decl @NoFields {}

// COM: Types from the standard library.
kgen.struct.decl @Error {}
kgen.struct.decl @Int {}

// CHECK-LABEL: @raises_error
lit.func @raises_error(%raise: i1, %err: !kgen.declref<@Error>, %value: !kgen.declref<@Int>) -> !pop.variant<@Error, @Int> {
  hlcf.if %raise {
    // CHECK: %[[ERR:.*]] = pop.variant.create %err
    %result = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, @Int>
    // CHECK: hlcf.return %[[ERR]]
    hlcf.return %result : !pop.variant<@Error, @Int>
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
    %raise = kgen.param.constant: i1 = <1>
    %result = kgen.call @raises_error(%raise, %err, %int)
      : (i1, !kgen.declref<@Error>, !kgen.declref<@Int>) -> !pop.variant<@Error, @Int>
    // CHECK: %[[VAL:.*]] = lit.unwrap_or_propagate %{{.*}} : <@Error, @Int>
    %value = lit.unwrap_or_propagate %result : <@Error, @Int>
    // CHECK: return %[[VAL]] : !kgen.declref<@Int>
    hlcf.return %value : !kgen.declref<@Int>
  // CHECK-NEXT: } except (%{{.*}}: !kgen.declref<@Error>) {
  } except (%exception: !kgen.declref<@Error>) {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: }
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
  // CHECK: kgen.struct.decl @A
  kgen.struct.decl @A {}

  // CHECK: kgen.struct.decl @B
  kgen.struct.decl @B {
    // CHECK-NEXT: lit.func @foo(%{{.*}}: !kgen.declref<@module::@B>, %{{.*}}: !pop.pointer<@module::@A>
    lit.func @foo(%self: !kgen.declref<@module::@B>, %a: !pop.pointer<@module::@A>) {
      kgen.return
    }
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !pop.pointer<@module::@A>, %b: !kgen.declref<@module::@B>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@module::@B>, !pop.pointer<@module::@A>) -> (): @module::@B::@foo]
  kgen.call_param[(!kgen.declref<@module::@B>, !pop.pointer<@module::@A>) -> (): @module::@B::@foo](%b, %a)
  kgen.return
}
