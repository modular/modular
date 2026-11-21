// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" --kgen-print-inline-type-values | FileCheck %s

!inner = !kgen.struct<(i32, i64)>
!outer = !kgen.struct<(!inner, i8)>
!array = !pop.array<320, index>

// Both GEPs return the same address, with different pointee types. Write
// through the field pointer, then load through the struct pointer.
kgen.generator @same_address(%initial: !outer, %replacement: i32) -> !inner {
  %storage = pop.stack_allocation 1 x !outer
  pop.store %initial, %storage : !kgen.pointer<!outer>
  %pair = kgen.struct.gep %storage[0] : <!outer>
  %field = kgen.struct.gep %pair[0] : <!inner>
  pop.store %replacement, %field : !kgen.pointer<i32>
  %result = pop.load %pair : !kgen.pointer<!inner>
  hlcf.return %result : !inner
}

// Populate more than 256 distinct addresses, then revisit every address.
kgen.generator @many_addresses(%end: index) -> index {
  %array = pop.stack_allocation 1 x !array
  %zero = index.constant 0
  %one = index.constant 1
  hlcf.loop (%i = %zero : index) {
    %done = index.cmp eq(%i, %end)
    %cond = pop.cast_from_builtin %done : i1 to !kgen.scalar<bool>
    hlcf.if %cond {
      hlcf.break
    } else {
      hlcf.yield
    }
    %idx = pop.cast_from_builtin %i : index to !kgen.scalar<index>
    %element = pop.array.gep %array[%idx] : <!array>
    pop.store %i, %element : !kgen.pointer<index>
    %next = index.add %i, %one
    hlcf.continue %next : index
  }
  %result = hlcf.loop (%sum = %zero : index, %i = %zero : index) -> index {
    %done = index.cmp eq(%i, %end)
    %cond = pop.cast_from_builtin %done : i1 to !kgen.scalar<bool>
    hlcf.if %cond {
      hlcf.break %sum : index
    } else {
      hlcf.yield
    }
    %idx = pop.cast_from_builtin %i : index to !kgen.scalar<index>
    %element = pop.array.gep %array[%idx] : <!array>
    %value = pop.load %element : !kgen.pointer<index>
    %next_sum = index.add %sum, %value
    %next = index.add %i, %one
    hlcf.continue %next_sum, %next : index, index
  }
  hlcf.return %result : index
}

// CHECK-LABEL: kgen.func export @test
kgen.generator export @test() {
  // CHECK-NEXT: kgen.param.constant: {{.*}} = <{ 71, 123456789 }>
  kgen.param.constant: !inner = <apply(
      :(!outer, i32) -> !inner @same_address, {{ 5, 123456789 }, 9}, 71)>
  // CHECK-NEXT: <51040>
  kgen.param.constant = <apply(:(index) -> index @many_addresses, 320)>
  hlcf.return
}
