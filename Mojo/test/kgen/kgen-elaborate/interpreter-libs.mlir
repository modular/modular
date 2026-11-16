// RUN: kgen %s -elaborate -S -o - --interpreter-libs=%interpreter_lib | FileCheck %s
// RUN: not kgen %s -elaborate -S -o - 2>&1 | FileCheck %s --check-prefix=NO-LIB
// RUN: rm -f %t.so && cp %interpreter_lib %t.so
// RUN: kgen %s -elaborate -S -o - --interpreter-libs=%t.so | FileCheck %s
// RUN: rm -f %t.v2.so && cp %interpreter_lib %t.v2.so
// RUN: env INTERPRETER_LIB_OFFSET=1 kgen %s -elaborate -S -o - --interpreter-libs=%t.v2.so | FileCheck %s --check-prefix=NEW-NAME
// RUN: rm -f %t.so && cp %s %t.so
// RUN: not kgen %s -elaborate -S -o - --interpreter-libs=%t.so 2>&1 | FileCheck %s --check-prefix=CHANGED-LIB
// RUN: not kgen %s -elaborate -S -o - --interpreter-libs=%t.missing 2>&1 | FileCheck %s --check-prefix=MISSING-LIB

// COM: `@sum` passes a cell that points at three values. The cell holds a
// COM: pointer, so the library function gets a copy that points at the values.
// COM: The values and the sum slot hold no pointers, so they are passed in
// COM: place and the library function's write to the sum slot is visible after
// COM: the call. The count and the result are scalar SIMD values, which travel
// COM: like integers. The cache
// COM: location depends on each library's file name and contents, so the runs
// COM: without a library, with a renamed one and with a rewritten one do not
// COM: reuse an earlier result. The renamed run sets INTERPRETER_LIB_OFFSET,
// COM: which the library reads, so a reused result would print the old value.

kgen.generator @sum(%count: index) -> index {
  %c1 = index.constant 1
  %c2 = index.constant 2
  %c10 = index.constant 10
  %c20 = index.constant 20
  %c30 = index.constant 30
  %c1000 = index.constant 1000
  %values = pop.stack_allocation 3 x index
  pop.store %c10, %values : !kgen.pointer<index>
  %off1 = pop.cast_from_builtin %c1 : index to !kgen.scalar<index>
  %v1 = pop.offset %values[%off1] : !kgen.pointer<index>
  pop.store %c20, %v1 : !kgen.pointer<index>
  %off2 = pop.cast_from_builtin %c2 : index to !kgen.scalar<index>
  %v2 = pop.offset %values[%off2] : !kgen.pointer<index>
  pop.store %c30, %v2 : !kgen.pointer<index>
  %cell = pop.stack_allocation 1 x !kgen.pointer<index>
  pop.store %values, %cell : !kgen.pointer<!kgen.pointer<index>>
  %out = pop.stack_allocation 1 x index
  %count_s = pop.cast_from_builtin %count : index to !kgen.scalar<index>
  %n_s = pop.external_call @sum_indirect(%cell, %count_s, %out)
    : (!kgen.pointer<!kgen.pointer<index>>, !kgen.scalar<index>, !kgen.pointer<index>) -> !kgen.scalar<index>
  %n = pop.cast_to_builtin %n_s : !kgen.scalar<index> to index
  %sum = pop.load %out : !kgen.pointer<index>
  %scaled = index.mul %sum, %c1000
  %result = index.add %scaled, %n
  hlcf.return %result : index
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> index {
  // CHECK: kgen.param.constant{{.*}} = <60003>
  // NEW-NAME: kgen.param.constant{{.*}} = <60004>
  %0 = kgen.param.constant = <apply(:(index) -> index @sum, 3)>
  hlcf.return %0 : index
}

// NO-LIB: unable to interpret call to unknown external function: sum_indirect
// CHANGED-LIB: cannot load interpreter library '{{.*}}.so'
// MISSING-LIB: cannot read interpreter library '{{.*}}.missing'
