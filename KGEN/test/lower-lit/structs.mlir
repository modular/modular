// RUN: kgen-opt -lower-lit -split-input-file %s | FileCheck %s

lit.struct.decl @Adder<size> {
  // CHECK-LABEL: kgen.generator @"Adder::__add__"<size>(%arg0: !kgen.declref<@Adder<size = size>>) {
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.func @__add__(%self: !kgen.declref<@Adder<size = size>>)  {
    %0 = lit.varlet.decl "a", var = true, synth=false : <index>
    %one = index.constant 1
    pop.store %one, %0 : !pop.pointer<index>
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
  lit.func @foo<M>(%self: !kgen.declref<@A<N = N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !kgen.declref<@A<N = 1>>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@A<N = 1>>) -> index: @"A::foo"<1, 2>]
  %0 = kgen.call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<1, 2>](%a)
  kgen.return
}

// -----

lit.struct.decl @A {
}

// CHECK: kgen.generator @rhslitdeclref_no_params(%arg0: !kgen.declref<@A>) {
lit.func @rhslitdeclref_no_params(%x: !kgen.declref<@A>) {
  kgen.return
}

// -----

lit.struct.decl @A<b, c> {
}

// CHECK: kgen.generator @rhslitdeclref_params(%arg0: !kgen.declref<@A<b = 10, c = 11>>) {
lit.func @rhslitdeclref_params(%x: !kgen.declref<@A<b = 10, c = 11>>) {
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
 // CHECK-LABEL:  @"foo::f"() -> index {
  lit.func @f() -> index {
    // CHECK: kgen.param.declare
    lit.alias.decl C = <3>
    %0 = kgen.param.constant: index = <1>
    kgen.return %0 : index
  }
}
