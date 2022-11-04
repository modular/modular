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
// CHECK-NEXT: %x = lit.var.decl "x" : <scalar<ty>>
lit.func @vardecl<ty : dtype>() {
  %x = lit.var.decl "x": !pop.pointer<scalar<ty>>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @SomeStruct<ty: dtype> {
lit.struct.decl @SomeStruct<ty: dtype> {
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

// CHECK-LABEL: lit.struct.decl @A
lit.struct.decl @A {
  // CHECK-NEXT: lit.func @"A::foo"
  lit.func @"A::foo"(%self: !kgen.ref<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: lit.struct.decl @B
lit.struct.decl @B {
  // CHECK-NEXT: lit.func @"B::foo"
  lit.func @"B::foo"(%self: !kgen.ref<@B>, %a: !kgen.ref<@A>) {
    // CHECK-NEXT: call_param[(!kgen.ref<@A>) -> (): @A::@"A::foo"]
    kgen.call_param[(!kgen.ref<@A>) -> (): @A::@"A::foo"](%a)
    kgen.return
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !kgen.ref<@A>, %b: !kgen.ref<@B>) {
  // CHECK-NEXT: call_param[(!kgen.ref<@B>, !kgen.ref<@A>) -> (): @B::@"B::foo"]
  kgen.call_param[(!kgen.ref<@B>, !kgen.ref<@A>) -> (): @B::@"B::foo"](%b, %a)
  // CHECK-NEXT: constant: (!kgen.ref<@A>) -> () = <@A::@"A::foo">
  %0 = kgen.param.constant: (!kgen.ref<@A>) -> () = <@A::@"A::foo">
  kgen.return
}

// -----

// CHECK-LABEL: lit.struct.decl @A<N>
lit.struct.decl @A<N> {
  // CHECK-NEXT: lit.func @"A::foo"<M>
  lit.func @"A::foo"<M>(%self: !kgen.ref<@A<N = N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: lit.func @main
lit.func @main(%a: !kgen.ref<@A<N = 1>>) {
  // CHECK-NEXT: call_param[<N, M>(!kgen.ref<@A<N = N>>) -> index: @A::@"A::foo"]<N = 1, M = 2>
  %0 = kgen.call_param[<N, M>(!kgen.ref<@A<N = N>>) -> index: @A::@"A::foo"]<N = 1, M = 2>(%a)
  // CHECK-NEXT: call_param[(!kgen.ref<@A<N = 1>>) -> index: @A::@"A::foo"<N = 1, M = 2>]
  %1 = kgen.call_param[(!kgen.ref<@A<N = 1>>) -> index: @A::@"A::foo"<N = 1, M = 2>](%a)
  kgen.return
}
