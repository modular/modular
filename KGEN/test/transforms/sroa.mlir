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
  // CHECK-NEXT: %[[MEM1:.*]] = pop.stack_allocation 1 x !pop.struct<scalar<index>> 
  // CHECK-NEXT: %[[MEM2:.*]] = pop.stack_allocation 1 x !pop.struct<scalar<index>> 
  // CHECK-NEXT: %[[MEM3:.*]] = pop.stack_allocation 1 x !pop.struct<scalar<index>> 

  // Check mem2reg is able to eliminate what is left.
  // MEM2REG: %[[SCALAR1:.*]] = pop.struct.extract %[[ARG0:.*]][0] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
  // MEM2REG-NEXT: %[[SCALAR2:.*]] = pop.struct.extract %[[ARG0:.*]][1] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
  // MEM2REG-NEXT: %[[SCALAR3:.*]] = pop.struct.extract %[[ARG0:.*]][2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
  // MEM2REG-NEXT:  hlcf.loop {
  // MEM2REG-NEXT:    hlcf.loop "inlined_cf_scope" {
  // MEM2REG-NEXT:       %[[EXTRACT1:.*]] = pop.struct.extract %[[SCALAR3:.*]][0] : !pop.struct<scalar<index>>
  // MEM2REG-NEXT:       %[[EXTRACT2:.*]] = pop.struct.extract %[[SCALAR1:.*]][0] : !pop.struct<scalar<index>>
  // MEM2REG-NEXT:       pop.div %[[EXTRACT2]], %[[EXTRACT1]] : !pop.scalar<index>

  kgen.return
}
