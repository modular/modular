// RUN: kgen-opt -lower-lit -split-input-file %s | FileCheck %s

// CHECK-LABEL: kgen.struct.decl @AdderOneField {
// CHECK-NEXT:    base : index
// CHECK-NEXT:  }
kgen.struct.decl @AdderOneField {
  %base = lit.var.decl "base" : <index>
}

kgen.struct.decl @Adder<size> {
  %base = lit.var.decl "base" : <index>

  // CHECK-LABEL: kgen.generator @"Adder::__add__"<size>(%arg0: !kgen.declref<@Adder<size = size>>) {
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.func @__add__(%self: !kgen.declref<@Adder<size = size>>)  {
    %0 = lit.var.decl "a" : <index>
    %one = index.constant 1
    pop.store %one, %0 : !pop.pointer<index>
    kgen.return
  }
}

// CHECK-LABEL: kgen.struct.decl @Adder<size> {
// CHECK-NEXT:    base : index
// CHECK-NEXT:  }

// -----

// CHECK-LABEL: kgen.generator @"A::foo"

// CHECK-LABEL: kgen.struct.decl @A
kgen.struct.decl @A {
  lit.func @foo(%self: !kgen.declref<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @"B::foo"
// CHECK-NEXT: call_param[(!kgen.declref<@A>) -> (): @"A::foo"]

// CHECK-LABEL: kgen.struct.decl @B
kgen.struct.decl @B {
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

// CHECK-LABEL: kgen.struct.decl @A<N>
kgen.struct.decl @A<N> {
  lit.func @foo<M>(%self: !kgen.declref<@A<N = N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !kgen.declref<@A<N = 1>>) {
  // CHECK-NEXT: call_param[(!kgen.declref<@A<N = 1>>) -> index: @"A::foo"<N = 1, M = 2>]
  %0 = kgen.call_param[(!kgen.declref<@A<N = 1>>) -> index: @A::@foo<N = 1, M = 2>](%a)
  kgen.return
}

// -----

kgen.struct.decl @A {
   %x = lit.var.decl "x" : <index>
 }

// CHECK: kgen.generator @rhslitdeclref_no_params(%arg0: !kgen.declref<@A>) -> !kgen.list<i1[0]> {
 lit.func @rhslitdeclref_no_params(%x: !kgen.declref<@A>) -> !lit.none {
  // CHECK: kgen.param.constant: list<i1[0]> = <[]>
   %0 = kgen.param.constant: !lit.none = <#lit.none>
   kgen.return %0 : !lit.none
 }

// -----

kgen.struct.decl @A<b, c> {
  %x = lit.var.decl "x" : <index>
}
// CHECK: kgen.generator @rhslitdeclref_params(%arg0: !kgen.declref<@A<b = 10, c = 11>>) -> !kgen.list<i1[0]> {
lit.func @rhslitdeclref_params(%x: !kgen.declref<@A<b = 10, c = 11>>) -> !lit.none {
  // CHECK: kgen.param.constant: list<i1[0]> = <[]>
  %0 = kgen.param.constant: !lit.none = <#lit.none>
  kgen.return %0 : !lit.none
}

// -----

kgen.struct.decl @A {
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
