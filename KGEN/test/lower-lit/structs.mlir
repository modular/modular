// RUN: kgen-opt -lower-lit %s | FileCheck %s

// CHECK-LABEL: kgen.struct.decl @AdderOneField {
// CHECK-NEXT:    base : index
// CHECK-NEXT:  }
lit.struct.decl @AdderOneField {
  %base = lit.var.decl "base" : <index>
}

lit.struct.decl @Adder<size> {
  %base = lit.var.decl "base" : <index>

  // CHECK-LABEL: kgen.generator @"Adder::__add__"(%arg0: !kgen.ref<@Adder<size = 2>>) {
  // CHECK-NEXT:    %[[ONE:.*]] = pop.stack_allocation 1 x index
  // CHECK:       }
  lit.func @"Adder::__add__"(%self: !kgen.ref<@Adder<size = 2>>)  {
    %0 = lit.var.decl "a" : <index>
    %one = index.constant 1
    pop.store %one, %0 : !pop.pointer<index>
    kgen.return
  }
}

// CHECK-LABEL: kgen.struct.decl @Adder<size> {
// CHECK-NEXT:    base : index
// CHECK-NEXT:  }
