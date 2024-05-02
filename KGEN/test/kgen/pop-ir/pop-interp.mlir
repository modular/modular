// RUN: kgen-opt %s -verify-parameters=simplify=true | FileCheck %s

kgen.generator @symbolic_stack_memory(%arg0: index) -> index {
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.generator @interpret
kgen.generator @interpret() {
  // CHECK-NEXT: constant = <42>
  kgen.param.constant = <apply(:(index) -> index @symbolic_stack_memory, 42)>
  kgen.return
}

kgen.generator @load_it(%arg0: !kgen.pointer<index>) -> index {
  %0 = pop.load %arg0 : !kgen.pointer<index>
  kgen.return %0 : index
}

// CHECK-LABEL: @load_from_store
kgen.generator @load_from_store() {
  // CHECK-NEXT: constant = <42>
  kgen.param.constant = <apply(:(!kgen.pointer<index>) -> index @load_it, store_to_mem(42))>
  kgen.return
}

kgen.generator @load_undef() -> index {
  %0 = pop.stack_allocation 1 x index
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: @interpret_undef
kgen.generator @interpret_undef() {
  // CHECK-NEXT: constant = <*?>
  kgen.param.constant = <apply(:() -> index @load_undef)>
  kgen.return
}
