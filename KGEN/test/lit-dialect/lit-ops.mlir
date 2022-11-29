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
  // CHECK-NEXT: call_param[<N, M>(!kgen.declref<@A<N = N>>) -> index: @A::@foo]<N = 1, M = 2>
  %0 = kgen.call_param[<N, M>(!kgen.declref<@A<N = N>>) -> index: @A::@foo]<N = 1, M = 2>(%a)
  // CHECK-NEXT: call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<N = 1, M = 2>]
  %1 = kgen.call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<N = 1, M = 2>](%a)
  kgen.return
}

// CHECK-LABEL: kgen.struct.decl @NoFields {
// CHECK-NEXT: }
kgen.struct.decl @NoFields {}
