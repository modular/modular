// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file %s | FileCheck %s

// A variable outside the default address space becomes a `pop.var_alloc`,
// which the target lowers after elaboration. Its lifetime markers are dropped
// because they only apply to stack slots.
// CHECK-LABEL: kgen.generator @shared_var
// CHECK-NEXT: %[[VAR:.*]] = pop.var_alloc "a" index address_space 3
// CHECK-NOT: lifetime
// CHECK: pop.store {{.*}}, %[[VAR]]
// CHECK-NEXT: hlcf.return
lit.fn @shared_var() {
  %a = lit.var.decl "a" var : !lit.ref<index, mut *"a", 3>
  lit.var.lifetime.start %a : !lit.ref<index, mut *"a", 3>
  %c = kgen.param.constant: index = <0>
  lit.ref.store %c, %a : <index, mut *"a", 3>
  lit.var.lifetime.end %a : !lit.ref<index, mut *"a", 3>
  hlcf.return
}

// -----

// A variable in the default address space keeps its stack slot and markers.
// CHECK-LABEL: kgen.generator @stack_var
// CHECK: %[[SLOT:.*]] = pop.stack_allocation 1 x index
// CHECK: pop.stack_alloc.lifetime.start(%[[SLOT]])
// CHECK: pop.stack_alloc.lifetime.end(%[[SLOT]])
lit.fn @stack_var() {
  %a = lit.var.decl "a" var : !lit.ref<index, mut *"a">
  lit.var.lifetime.start %a : !lit.ref<index, mut *"a">
  lit.var.lifetime.end %a : !lit.ref<index, mut *"a">
  hlcf.return
}
