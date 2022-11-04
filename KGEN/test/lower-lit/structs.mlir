// RUN: kgen-opt -lower-lit -split-input-file %s | FileCheck %s

// CHECK-LABEL: kgen.struct.decl @AdderOneField {
// CHECK-NEXT:    base : index
// CHECK-NEXT:  }
lit.struct.decl @AdderOneField {
  %base = lit.var.decl "base" : <index>
}

lit.struct.decl @Adder<size> {
  %base = lit.var.decl "base" : <index>

  // CHECK-LABEL: kgen.generator @"Adder::__add__"<size>(%arg0: !kgen.ref<@Adder<size = size>>) {
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.func @"Adder::__add__"(%self: !kgen.ref<@Adder<size = size>>)  {
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
lit.struct.decl @A {
  lit.func @"A::foo"(%self: !kgen.ref<@A>) {
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @"B::foo"
// CHECK-NEXT: call_param[(!kgen.ref<@A>) -> (): @"A::foo"]

// CHECK-LABEL: kgen.struct.decl @B
lit.struct.decl @B {
  lit.func @"B::foo"(%self: !kgen.ref<@B>, %a: !kgen.ref<@A>) {
    kgen.call_param[(!kgen.ref<@A>) -> (): @A::@"A::foo"](%a)
    kgen.return
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !kgen.ref<@A>, %b: !kgen.ref<@B>) {
  // CHECK-NEXT: call_param[(!kgen.ref<@B>, !kgen.ref<@A>) -> (): @"B::foo"]
  kgen.call_param[(!kgen.ref<@B>, !kgen.ref<@A>) -> (): @B::@"B::foo"](%b, %a)
  // CHECK-NEXT: constant: (!kgen.ref<@A>) -> () = <@"A::foo">
  %0 = kgen.param.constant: (!kgen.ref<@A>) -> () = <@A::@"A::foo">
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @"A::foo"<N, M>

// CHECK-LABEL: kgen.struct.decl @A<N>
lit.struct.decl @A<N> {
  lit.func @"A::foo"<M>(%self: !kgen.ref<@A<N = N>>) -> index {
    %0 = kgen.param.constant = <add(N, M)>
    kgen.return %0 : index
  }
}

// CHECK-LABEL: kgen.generator @main
lit.func @main(%a: !kgen.ref<@A<N = 1>>) {
  // CHECK-NEXT: call_param[<N, M>(!kgen.ref<@A<N = N>>) -> index: @"A::foo"]<N = 1, M = 2>
  %0 = kgen.call_param[<N, M>(!kgen.ref<@A<N = N>>) -> index: @A::@"A::foo"]<N = 1, M = 2>(%a)
  // CHECK-NEXT: call_param[(!kgen.ref<@A<N = 1>>) -> index: @"A::foo"<N = 1, M = 2>]
  %1 = kgen.call_param[(!kgen.ref<@A<N = 1>>) -> index: @A::@"A::foo"<N = 1, M = 2>](%a)
  kgen.return
}
