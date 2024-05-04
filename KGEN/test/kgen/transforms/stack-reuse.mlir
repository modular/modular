// RUN: kgen-opt %s -stack-reuse | FileCheck %s

// CHECK-LABEL: @two_overlapping
kgen.func @two_overlapping(%arg0: index, %arg1: index) -> (index, index) {
  // CHECK-NEXT: %[[S0:.*]] = pop.stack_allocation
  // CHECK-NEXT: %[[S1:.*]] = pop.stack_allocation
  %s0 = pop.stack_allocation 1 x index
  %s1 = pop.stack_allocation 1 x index
  pop.store %arg0, %s0 : !kgen.pointer<index>
  pop.store %arg1, %s1 : !kgen.pointer<index>

  // CHECK-NOT: pop.stack_allocation
  %s2 = pop.stack_allocation 1 x index
  pop.store %arg0, %s2 : !kgen.pointer<index>
  // CHECK: %[[V0:.*]] = pop.load %[[S0]]
  %v0 = pop.load %s2 : !kgen.pointer<index>
  pop.store %arg1, %s2 : !kgen.pointer<index>
  // CHECK: %[[V1:.*]] = pop.load %[[S1]]
  %v1 = pop.load %s2 : !kgen.pointer<index>
  // CHECK: return %[[V0]], %[[V1]]
  kgen.return %v0, %v1 : index, index
}

// CHECK-LABEL: @control_flow_if
kgen.func @control_flow_if(%arg0: index, %arg1: index, %arg2: i1) -> index {
  // CHECK-NEXT: %[[S0:.*]] = pop.stack_allocation
  // CHECK-NOT: pop.stack_allocation
  %s0 = pop.stack_allocation 1 x index
  %s1 = pop.stack_allocation 1 x index

  pop.store %arg0, %s0 : !kgen.pointer<index>
  // CHECK: hlcf.if
  %if = hlcf.if %arg2 -> index {
    // CHECK-NEXT: pop.load
    %0 = pop.load %s0 : !kgen.pointer<index>
    pop.store %0, %s1 : !kgen.pointer<index>
    // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
    %1 = pop.load %s1 : !kgen.pointer<index>
    // CHECK-NEXT: yield %[[V]]
    hlcf.yield %1 : index
  // CHECK: else
  } else {
    // CHECK-NEXT: pop.load
    %0 = pop.load %s0 : !kgen.pointer<index>
    pop.store %0, %s1 : !kgen.pointer<index>
    // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
    %1 = pop.load %s1 : !kgen.pointer<index>
    // CHECK-NEXT: yield %[[V]]
    hlcf.yield %1 : index
  }
  // CHECK: pop.load
  %0 = pop.load %s0 : !kgen.pointer<index>
  pop.store %0, %s1 : !kgen.pointer<index>
  // CHECK-NEXT: %[[V:.*]] = pop.load %[[S0]]
  %1 = pop.load %s1 : !kgen.pointer<index>
  // CHECK-NEXT: return %[[V]]
  kgen.return %1 : index
}

// CHECK-LABEL: @loop_and_gep
kgen.func @loop_and_gep(%arg0: !pop.array<2, index>, %arg1: index) {
  // CHECK-NEXT: %[[S0:.*]] = pop.stack_allocation
  %s0 = pop.stack_allocation 1 x !pop.array<2, index>
  pop.store %arg0, %s0 : !kgen.pointer<array<2, index>>
  // CHECK: hlcf.loop
  hlcf.loop {
    %0 = pop.load %s0 : !kgen.pointer<array<2, index>>
    // CHECK-NOT: pop.stack_allocation
    %s1 = pop.stack_allocation 1 x !pop.array<2, index>
    pop.store %0, %s1 : !kgen.pointer<array<2, index>>
    // CHECK: %[[GEP:.*]] = pop.array.gep %[[S0]][%arg1]
    %1 = pop.array.gep %s1[%arg1] : <array<2, index>>
    // CHECK-NEXT: pop.load %[[GEP]]
    %2 = pop.load %1 : !kgen.pointer<index>
    hlcf.continue
  }
  // CHECK: pop.load
  %0 = pop.load %s0 : !kgen.pointer<array<2, index>>
  // CHECK-NOT: pop.stack_allocation
  %s1 = pop.stack_allocation 1 x !pop.array<2, index>
  pop.store %0, %s1 : !kgen.pointer<array<2, index>>
  // CHECK: %[[GEP:.*]] = pop.array.gep %[[S0]][%arg1]
  %1 = pop.array.gep %s1[%arg1] : <array<2, index>>
  // CHECK-NEXT: pop.load %[[GEP]]
  %2 = pop.load %1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: @gep_reconstruct
kgen.func @gep_reconstruct(%arg0: !pop.array<2, index>, %arg1: !pop.array<2, index>) -> (index, index) {
  %idx0 = index.constant 0

  // CHECK: %[[S0:.*]] = pop.stack_allocation
  %s0 = pop.stack_allocation 1 x !pop.array<2, index>
  // CHECK-NEXT: %[[S1:.*]] = pop.stack_allocation
  %s1 = pop.stack_allocation 1 x !pop.array<2, index>

  pop.store %arg0, %s0 : !kgen.pointer<array<2, index>>
  pop.store %arg1, %s1 : !kgen.pointer<array<2, index>>

  // CHECK-NOT: pop.stack_allocation
  %s2 = pop.stack_allocation 1 x !pop.array<2, index>
  %gep = pop.array.gep %s2[%idx0] : <array<2, index>>

  pop.store %arg0, %s2 : !kgen.pointer<array<2, index>>
  // CHECK: %[[GEP0:.*]] = pop.array.gep %[[S0]][%idx0]
  // CHECK-NEXT: %[[R0:.*]] = pop.load %[[GEP0]]
  %r0 = pop.load %gep : !kgen.pointer<index>
  pop.store %arg1, %s2 : !kgen.pointer<array<2, index>>
  // CHECK: %[[GEP1:.*]] = pop.array.gep %[[S1]][%idx0]
  // CHECK-NEXT: %[[R1:.*]] = pop.load %[[GEP1]]
  %r1 = pop.load %gep : !kgen.pointer<index>

  // CHECK-NEXT: return %[[R0]], %[[R1]]
  kgen.return %r0, %r1 : index, index
}

// CHECK-LABEL: @no_alloc_users
kgen.func @no_alloc_users() {
  hlcf.loop {
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @use_in_region
kgen.func @use_in_region(%arg0 : index) {
  // CHECK-NEXT: %0 = pop.stack_allocation
  %0 = pop.stack_allocation 1 x index
  // CHECK-NOT: pop.stack_allocation
  %1 = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.store %arg0, %0
  pop.store %arg0, %0 : !kgen.pointer<index>
  pop.store %arg0, %1 : !kgen.pointer<index>
  // CHECK-NEXT: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: pop.load %0
    %2 = pop.load %1 : !kgen.pointer<index>
    // CHECK-NEXT: pop.load %0
    %3 = pop.load %0 : !kgen.pointer<index>
    // CHECK-NEXT: hlcf.break
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @use_crosses_region
kgen.func @use_crosses_region(%arg0: index) {
  // CHECK-NEXT: %0 = pop.stack_allocation
  %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: %1 = pop.stack_allocation
  %1 = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.store %arg0, %0
  pop.store %arg0, %0 : !kgen.pointer<index>
  // CHECK-NEXT: pop.store %arg0, %1
  pop.store %arg0, %1 : !kgen.pointer<index>
  // CHECK-NEXT: stage_closure
  kgen.stage_closure = () -> () {
    // CHECK-NEXT: pop.load %1
    pop.load %1 : !kgen.pointer<index>
    // CHECK-NEXT: pop.store %arg0, %0
    pop.store %arg0, %0 : !kgen.pointer<index>
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: @copy_elision_alias
// TODO(#22921): The pass should be able to elide this. Just make sure it
// doesn't crash for now.
kgen.func @copy_elision_alias() {
  %0 = pop.stack_allocation 1 x struct<(struct<(index)>)>
  %1 = kgen.struct.gep %0[0] : <struct<(struct<(index)>)>>
  %2 = pop.load %1 : !kgen.pointer<struct<(index)>>
  %3 = pop.stack_allocation 1 x struct<(index)>
  pop.store %2, %3 : !kgen.pointer<struct<(index)>>
  pop.load %3 : !kgen.pointer<struct<(index)>>
  kgen.return
}

// CHECK-LABEL: @function_boundary
kgen.func @function_boundary(%arg0: index) -> index {
  // CHECK: [[S0:%.*]] = pop.stack_allocation
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !kgen.pointer<index>
  %1 = pop.stack_allocation 1 x index
  pop.store %arg0, %1 : !kgen.pointer<index>
  // CHECK: lit.async.execute
  lit.async.execute : index {
    // CHECK: [[S1:%.*]] = pop.stack_allocation
    %3 = pop.stack_allocation 1 x index
    pop.store %arg0, %3 : !kgen.pointer<index>
    %4 = pop.stack_allocation 1 x index
    pop.store %arg0, %4 : !kgen.pointer<index>
    // CHECK: [[R1:%.*]] = pop.load [[S1]]
    %5 = pop.load %4 : !kgen.pointer<index>
    // CHECK-NEXT: return [[R1]]
    kgen.return %5 : index
  }
  // CHECK: [[R0:%.*]] = pop.load [[S0]]
  %2 = pop.load %1 : !kgen.pointer<index>
  // CHECK-NEXT: return [[R0]]
  kgen.return %2 : index
}
