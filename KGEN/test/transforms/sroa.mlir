// RUN: kgen-opt -sroa -allow-unregistered-dialect %s | FileCheck %s

// Check sroa runs as expected along side mem-2-reg
// RUN: kgen-opt -sroa -mem-2-reg -allow-unregistered-dialect %s | FileCheck -check-prefix="MEM2REG" %s

// CHECK-LABEL: @simple_struct
// MEM2REG-LABEL: @simple_struct
kgen.func @simple_struct(%arg1: !pop.struct<index, index>) -> !pop.scalar<index> {
  %array = pop.stack_allocation 1 x !pop.struct<index, index>
  pop.store %arg1, %array : !pop.pointer<struct<index, index>>

  // CHECK: %[[MEM1:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x index

  // Extract from the input and store into stack.
  // CHECK-NEXT: %[[EXTRACT:.*]] = pop.struct.extract %[[ARG0:.*]][0] : !pop.struct<index, index>
  // CHECK-NEXT: pop.store %[[EXTRACT]], %[[MEM1]] : !pop.pointer<index>
  // CHECK-NEXT: %[[EXTRACT2:.*]] = pop.struct.extract %[[ARG0]][1] : !pop.struct<index, index>
  // CHECK-NEXT: pop.store %[[EXTRACT2]], %[[MEM2]] : !pop.pointer<index>


  // Load from stack.
  // CHECK-NEXT: pop.load %[[MEM1]] : !pop.pointer<index>
  // CHECK-NEXT: pop.load %[[MEM2]] : !pop.pointer<index>


  // When running with mem2reg check we get rid of the allocs.
  // MEM2REG-NEXT: %[[SCALAR1:.*]] = pop.struct.extract %[[ARG0:.*]][0] : !pop.struct<index, index>
  // MEM2REG-NEXT: %[[SCALAR2:.*]] =  pop.struct.extract %[[ARG0]][1] : !pop.struct<index, index>
  // MEM2REG-NEXT: pop.cast_from_builtin %[[SCALAR1]] : index to !pop.scalar<index>
  // MEM2REG-NEXT: pop.cast_from_builtin %[[SCALAR2]] : index to !pop.scalar<index>

  %gep1 = pop.struct.gep %array[0] : <struct<index, index>>
  %gep2 = pop.struct.gep %array[1] : <struct<index, index>>

  %load1 = pop.load %gep1 : !pop.pointer<index>
  %load2 = pop.load %gep2 : !pop.pointer<index>
  %scalar1 = pop.cast_from_builtin %load1 : index to !pop.scalar<index>
  %scalar2 = pop.cast_from_builtin %load2 : index to !pop.scalar<index>
  %out = pop.add %scalar1, %scalar2 : !pop.scalar<index>
  kgen.return %out : !pop.scalar<index>
}

// CHECK-LABEL: @simple_array
// MEM2REG-LABEL: @simple_array
kgen.func @simple_array(%arg1: !pop.array<2, index>) -> !pop.scalar<index> {
   %0 = kgen.param.constant = <0>
   %1 = kgen.param.constant = <1>

  // CHECK: %[[MEM1:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x index

  // Extract from the input and store into stack.
  // CHECK-NEXT: %[[GET:.*]] = pop.array.get %[[ARG0:.*]][0] : !pop.array<2, index>
  // CHECK-NEXT: pop.store %[[GET]], %[[MEM1]] : !pop.pointer<index>
  // CHECK-NEXT: %[[GET2:.*]] = pop.array.get %[[ARG0]][1] : !pop.array<2, index>
  // CHECK-NEXT: pop.store %[[GET2]], %[[MEM2]] : !pop.pointer<index>


  // Load from stack.
  // CHECK-NEXT: pop.load %[[MEM1]] : !pop.pointer<index>
  // CHECK-NEXT: pop.load %[[MEM2]] : !pop.pointer<index>

  // MEM2REG: %[[OP1:.*]] = pop.array.get %[[ARG0:.*]][0] : !pop.array<2, index>
  // MEM2REG-NEXT: %[[OP2:.*]] = pop.array.get %[[ARG0]][1] : !pop.array<2, index>
  // MEM2REG-NEXT: %[[CONVERT1:.*]] = pop.cast_from_builtin %[[OP1]] : index to !pop.scalar<index>
  // MEM2REG-NEXT: %[[CONVERT2:.*]] = pop.cast_from_builtin %[[OP2]] : index to !pop.scalar<index>
  // MEM2REG-NEXT: %[[ADD:.*]] = pop.add %[[CONVERT1]], %[[CONVERT2]] : !pop.scalar<index>
  // MEM2REG-NEXT: kgen.return %[[ADD]] : !pop.scalar<index>

   %array = pop.stack_allocation 1 x !pop.array<2, index>
   pop.store %arg1, %array : !pop.pointer<array<2, index>>

   %gep1 = pop.array.gep %array[%0] : <array<2, index>>
   %gep2 = pop.array.gep %array[%1] : <array<2, index>>

   %load1 = pop.load %gep1 : !pop.pointer<index>
   %load2 = pop.load %gep2 : !pop.pointer<index>
   %scalar1 = pop.cast_from_builtin %load1 : index to !pop.scalar<index>
   %scalar2 = pop.cast_from_builtin %load2 : index to !pop.scalar<index>
   %out = pop.add %scalar1, %scalar2 : !pop.scalar<index>
   kgen.return %out : !pop.scalar<index>
 }

// CHECK-LABEL: @struct_of_structs
// MEM2REG-LABEL: @struct_of_structs
kgen.func @struct_of_structs(%arg1: !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>) {
  %memory = pop.stack_allocation 1 x !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
  pop.store %arg1, %memory : !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>
  hlcf.loop {
    %load = pop.load %memory : !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>
    hlcf.loop "inlined_cf_scope" {
      %getElem1 = pop.struct.extract %load[2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
      %getElem2 = pop.struct.extract %getElem1[0] : !pop.struct<scalar<index>>

      %gep = pop.struct.gep %memory[0] : <struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>
      %newLoad = pop.load %gep : !pop.pointer<struct<scalar<index>>>
      %getElem3 = pop.struct.extract %newLoad[0] : !pop.struct<scalar<index>>

      %out = pop.div %getElem3, %getElem2 : !pop.scalar<index>
      hlcf.break
    }
    hlcf.break
  }

  // Just check this has been broken into several allocations.
  // CHECK-NEXT: %[[MEM1:.*]] = pop.stack_allocation 1 x !pop.scalar<index>
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x !pop.scalar<index>
  // CHECK-NEXT: %[[MEM3:.*]] = pop.stack_allocation 1 x !pop.scalar<index>

  // In this test the incoming argument is a struct of structs so we can't
  // sroa the argument. Still check that we are still left with no stack alloc
  // and that all the innerloop uses are of the fully extracted base type.

  // MEM2REG: %[[OP0:.*]] = pop.struct.extract %[[ARG0:.*]][0] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
  // MEM2REG-DAG: %[[OP1:.*]] = pop.struct.extract %[[OP0]][0] : !pop.struct<scalar<index>>
  // MEM2REG-DAG: %[[OP2:.*]] = pop.struct.extract %[[ARG0]][1] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
  // MEM2REG-DAG: %[[OP3:.*]] = pop.struct.extract %[[OP2]][0] : !pop.struct<scalar<index>>
  // MEM2REG-DAG: %[[OP4:.*]] = pop.struct.extract %[[ARG0]][2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
  // MEM2REG-DAG: %[[OP5:.*]] = pop.struct.extract %[[OP4]][0] : !pop.struct<scalar<index>>
  // MEM2REG-DAG:  hlcf.loop {
  // MEM2REG-DAG:    hlcf.loop "inlined_cf_scope" {
  // MEM2REG-DAG:       pop.div %[[OP1]], %[[OP5]] : !pop.scalar<index>

  kgen.return
}

// CHECK-LABEL: @stack_of_N
// MEM2REG-LABEL: @stack_of_N
// CHECK: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[OUT_PTR:.*]]: !pop.pointer<index>)
// MEM2REG: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[OUT_PTR:.*]]: !pop.pointer<index>)
kgen.func @stack_of_N(%val1: index, %val2: index, %val3: index, %output : !pop.pointer<index>) {
  %0 = kgen.param.constant = <0>
  %1 = kgen.param.constant = <1>
  %2 = kgen.param.constant = <2>

  %alloc = pop.stack_allocation 3 x index

  // CHECK: %[[MEM1:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM3:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.store %[[ARG0]], %[[MEM1]] : !pop.pointer<index>
  // CHECK-NEXT: pop.store %[[ARG1]], %[[MEM2]] : !pop.pointer<index>
  // CHECK-NEXT: pop.store %[[ARG2]], %[[MEM3]] : !pop.pointer<index>

  // Mem2Reg should eliminate everything
  // MEM2REG-NEXT: kgen.param.constant = <0>
  // MEM2REG-NEXT: kgen.param.constant = <1>
  // MEM2REG-NEXT: kgen.param.constant = <2>
  // MEM2REG-NEXT: pop.store %[[ARG2]], %[[OUT_PTR]] : !pop.pointer<index>
  // MEM2REG-NEXT: kgen.return

  %offset1 = pop.offset %alloc[%0] : !pop.pointer<index>
  pop.store %val1, %offset1 : !pop.pointer<index>

  %offset2 = pop.offset %alloc[%1] : !pop.pointer<index>
  pop.store %val2, %offset2 : !pop.pointer<index>

  %offset3 = pop.offset %alloc[%2] : !pop.pointer<index>
  pop.store %val3, %offset3 : !pop.pointer<index>

  %annoying_offset = pop.offset %alloc[%2] : !pop.pointer<index>
  %load = pop.load %annoying_offset align 8  : !pop.pointer<index>
  pop.store %load, %output : !pop.pointer<index>
  kgen.return
}


// CHECK-LABEL: @bigger_stack
// CHECK: (%[[ARG0:.*]]: index, %[[OUT_PTR:.*]]: !pop.pointer<index>)
kgen.func @bigger_stack(%val1: index, %output : !pop.pointer<index>) {
  %0 = kgen.param.constant = <0>

  // Larger stacks should not be touched.
  // CHECK: pop.stack_allocation 32 x index

  %alloc = pop.stack_allocation 32 x index
  %offset = pop.offset %alloc[%0] : !pop.pointer<index>
  pop.store %val1, %offset : !pop.pointer<index>
  %load = pop.load %offset align 8  : !pop.pointer<index>
  pop.store %load, %output : !pop.pointer<index>
  kgen.return
}

// Handle storing directly to the stack as an implicit offset of 0.
// CHECK-LABEL: @n_stack_store
// MEM2REG-LABEL: @n_stack_store
// CHECK: (%[[ARG0:.*]]: index, %[[OUT_PTR:.*]]: !pop.pointer<index>)
// MEM2REG: (%[[ARG0:.*]]: index, %[[OUT_PTR:.*]]: !pop.pointer<index>)
kgen.func @n_stack_store(%val1: index, %output : !pop.pointer<index>) {
  %alloc = pop.stack_allocation 3 x index
  pop.store %val1, %alloc : !pop.pointer<index>
  %load = pop.load %alloc align 8  : !pop.pointer<index>
  pop.store %load, %output : !pop.pointer<index>

  // CHECK-NEXT: %[[MEM1:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: %[[MEM3:.*]] = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.store %[[ARG0]], %[[MEM1]] : !pop.pointer<index>
  // CHECK-NEXT: %[[LOAD:.*]] = pop.load %[[MEM1]] : !pop.pointer<index>
  // CHECK-NEXT: pop.store %[[LOAD]], %[[OUT_PTR]] : !pop.pointer<index>


  // Mem2Reg should eliminate everything
  // MEM2REG-NEXT: pop.store %[[ARG0]], %[[OUT_PTR]] : !pop.pointer<index>
  // MEM2REG-NEXT: kgen.return

  kgen.return
}

// CHECK-LABEL: @n_stack_arrays
// MEM2REG-LABEL: @n_stack_arrays
// CHECK: (%[[ARG0:.*]]: !pop.array<3, index>, %[[OUT_PTR:.*]]: !pop.pointer<index>)
// MEM2REG: (%[[ARG0:.*]]: !pop.array<3, index>, %[[OUT_PTR:.*]]: !pop.pointer<index>)
kgen.func @n_stack_arrays(%val: !pop.array<3, index>, %output : !pop.pointer<index>) {
  %0 = kgen.param.constant = <0>
  %1 = kgen.param.constant = <1>

  // CHECK: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index

  // Mem2reg should have enough information to realize they are aliases of the
  // same value.
  // MEM2REG: pop.array.get %[[ARG0]][0] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][1] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][2] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][0] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][1] : !pop.array<3, index>
  // MEM2REG-NEXT: pop.array.get %[[ARG0]][2] : !pop.array<3, index>

  %alloc = pop.stack_allocation 5 x !pop.array<3, index>
  pop.store %val, %alloc : !pop.pointer<array<3, index>>
  %offset = pop.offset %alloc[%1] : !pop.pointer<array<3, index>>
  pop.store %val, %offset : !pop.pointer<array<3, index>>

  %gep1 = pop.array.gep %alloc[%0] : <array<3, index>>
  %gep2 = pop.array.gep %offset[%1] : <array<3, index>>

  %load = pop.load %gep1 align 8  : !pop.pointer<index>
  pop.store %load, %output : !pop.pointer<index>

  %load2 = pop.load %gep2 align 8  : !pop.pointer<index>
  pop.store %load2, %output : !pop.pointer<index>

  kgen.return
}

// CHECK-LABEL: @n_stack_structs
// MEM2REG-LABEL: @n_stack_structs
// CHECK: (%[[ARG0:.*]]: !pop.struct<index, index>, %[[OUT_PTR:.*]]: !pop.pointer<index>)
// MEM2REG: (%[[ARG0:.*]]: !pop.struct<index, index>, %[[OUT_PTR:.*]]: !pop.pointer<index>)
kgen.func @n_stack_structs(%val: !pop.struct<index, index>, %output : !pop.pointer<index>) {
  %1 = kgen.param.constant = <1>

  %alloc = pop.stack_allocation 5 x !pop.struct<index, index>
  pop.store %val, %alloc : !pop.pointer<struct<index, index>>
  %offset = pop.offset %alloc[%1] : !pop.pointer<struct<index, index>>
  pop.store %val, %offset : !pop.pointer<struct<index, index>>

  // CHECK: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation 1 x index


  // MEM2REG: pop.struct.extract %[[ARG0]][0] : !pop.struct<index, index>
  // MEM2REG-NEXT: pop.struct.extract %[[ARG0]][1] : !pop.struct<index, index>
  // MEM2REG-NEXT: pop.struct.extract %[[ARG0]][0] : !pop.struct<index, index>
  // MEM2REG-NEXT: pop.struct.extract %[[ARG0]][1] : !pop.struct<index, index>

  %gep1 = pop.struct.gep %alloc[0] : <struct<index, index>>
  %gep2 = pop.struct.gep %offset[1] : <struct<index, index>>

  %load = pop.load %gep1 align 8  : !pop.pointer<index>
  pop.store %load, %output : !pop.pointer<index>

  %load2 = pop.load %gep2 align 8  : !pop.pointer<index>
  pop.store %load2, %output : !pop.pointer<index>

  kgen.return
}

// CHECK-LABEL: @store_arg
kgen.func @store_arg(
    %arg0: !pop.pointer<pointer<index>>,
    %arg1: !pop.pointer<pointer<struct<index>>>,
    %arg2: !pop.pointer<pointer<array<2, index>>>) {
  // CHECK: stack_allocation 2 x index
  %0 = pop.stack_allocation 2 x index
  pop.store %0, %arg0 : !pop.pointer<pointer<index>>
  // CHECK: stack_allocation 1 x !pop.struct<index>
  %1 = pop.stack_allocation 1 x !pop.struct<index>
  pop.store %1, %arg1 : !pop.pointer<pointer<struct<index>>>
  // CHECK: stack_allocation 1 x !pop.array<2, index>
  %2 = pop.stack_allocation 1 x !pop.array<2, index>
  pop.store %2, %arg2 : !pop.pointer<pointer<array<2, index>>>
  kgen.return
}

kgen.func @negArrayGep() {
  // CHECK: kgen.param.constant = <-1>
  // CHECK-NEXT: pop.stack_allocation 1 x !pop.array<2, index>
  // CHECK-NEXT: pop.array.gep
  %0 = kgen.param.constant = <-1>
  %array = pop.stack_allocation 1 x !pop.array<2, index>
  %gep = pop.array.gep %array[%0] : <array<2, index>>
  kgen.return
}


kgen.func @negOffsetGep() {
  // CHECK: kgen.param.constant = <-1>
  // CHECK-NEXT: pop.stack_allocation 2 x index
  // CHECK-NEXT: pop.offset
  %0 = kgen.param.constant = <-1>
  %alloc = pop.stack_allocation 2 x index
  %offset = pop.offset %alloc[%0] : !pop.pointer<index>
  kgen.return
}
