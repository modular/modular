// RUN: kgen-opt -mem-2-reg -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @simple_add
kgen.generator @simple_add(%arg0: index, %arg1: index) -> index {
  // CHECK-NEXT: %0 = index.add %arg0, %arg1
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !pop.pointer<index>

  %1 = pop.stack_allocation 1 x index
  pop.store %arg1, %1 : !pop.pointer<index>

  %2 = pop.load %0 : !pop.pointer<index>
  %3 = pop.load %1 : !pop.pointer<index>
  %4 = index.add %2, %3
  pop.store %4, %1 : !pop.pointer<index>

  %5 = pop.load %1 : !pop.pointer<index>
  // CHECK-NEXT: return %0
  kgen.return %5 : index
}

// CHECK-LABEL: @use_in_region
kgen.generator @use_in_region(%arg0: index, %arg1: i1) -> index {
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !pop.pointer<index>

  // CHECK-NEXT: hlcf.if
  hlcf.if %arg1 {
    // CHECK-NEXT: kgen.return %arg0
    %1 = pop.load %0 : !pop.pointer<index>
    kgen.return %1 : index
  } else {
    hlcf.yield
  }

  // CHECK-NOT: pop.load
  %1 = pop.load %0 : !pop.pointer<index>
  // CHECK: return %arg0 : index
  kgen.return %1 : index
}

// CHECK-LABEL: @store_in_region
kgen.generator @store_in_region(%arg0: index, %arg1: index, %arg2: i1) -> index {
  // CHECK-NEXT: stack_allocation
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !pop.pointer<index>

  hlcf.if %arg2 {
    %1 = pop.load %0 : !pop.pointer<index>
    kgen.return %1 : index
  } else {
    pop.store %arg1, %0 : !pop.pointer<index>
    hlcf.yield
  }

  %1 = pop.load %0 : !pop.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: @unknown_use
kgen.generator @unknown_use(%arg0: index) -> index {
  // CHECK-NEXT: stack_allocation
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !pop.pointer<index>
  "unknown.use"(%0) : (!pop.pointer<index>) -> ()
  %1 = pop.load %0 : !pop.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: @nested_alloc
kgen.generator @nested_alloc(%arg0: index) -> index {
  // CHECK-NEXT: hlcf.loop
  %0 = hlcf.loop () -> index {
    %1 = pop.stack_allocation 1 x index
    pop.store %arg0, %1 : !pop.pointer<index>
    // CHECK-NEXT: hlcf.loop
    %2 = hlcf.loop () -> index {
      %3 = pop.load %1 : !pop.pointer<index>
      // CHECK-NEXT: hlcf.break %arg0
      hlcf.break %3 : index
    }
    hlcf.break %2 : index
  }
  kgen.return %0 : index
}

// CHECK-LABEL: @read_uninitialized
kgen.generator @read_uninitialized() -> index {
  // CHECK-NEXT: %0 = kgen.undef : index
  %0 = pop.stack_allocation 1 x index
  %1 = pop.load %0 : !pop.pointer<index>
  // CHECK-NEXT: kgen.return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @if_empty_block
kgen.generator @if_empty_block(%arg0: i1, %arg1: index) -> index{
  // CHECK-NEXT: %0 = pop.stack_allocation
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x index
  pop.store %arg1, %0 : !pop.pointer<index>
  // CHECK-NEXT: hlcf.if
  hlcf.if %arg0 {
    // CHECK-NEXT: pop.store %arg1, %0
    %2 = pop.load %0 : !pop.pointer<index>
    pop.store %2, %1 : !pop.pointer<index>
    hlcf.yield
  } else {
    hlcf.yield
  }
  %2 = pop.load %1 : !pop.pointer<index>
  kgen.return %2 : index
}

// CHECK-LABEL: @store_alloca
kgen.func @store_alloca() -> i32 {
  // CHECK-NEXT: pop.stack_allocation 1 x i32
  // CHECK-NEXT: pop.load
  %0 = pop.stack_allocation 1 x !pop.pointer<i32>
  %1 = pop.stack_allocation 1 x i32
  pop.store %1, %0 : !pop.pointer<pointer<i32>>
  %2 = pop.load %0 : !pop.pointer<pointer<i32>>
  %3 = pop.load %2 : !pop.pointer<i32>
  kgen.return %3 : i32
}

// CHECK-LABEL: @try_region
kgen.func @try_region() {
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x index
  %idx2 = index.constant 2
  %idx3 = index.constant 3
  pop.store %idx2, %0 : !pop.pointer<index>
  pop.store %idx3, %1 : !pop.pointer<index>
  // CHECK: lit.try
  lit.try {
    // CHECK-NEXT: "use"(%idx2)
    %2 = pop.load %0 : !pop.pointer<index>
    "use"(%2) : (index) -> ()
    // CHECK-NEXT: pop.store
    pop.store %idx2, %1 : !pop.pointer<index>
    lit.try.yield
  // CHECK: except
  } except (%e: !pop.struct<>) {
    // CHECK-NEXT: pop.load
    %2 = pop.load %1 : !pop.pointer<index>
    "use"(%2) : (index) -> ()
    lit.try.yield
  // CHECK: else
  } else {
    // CHECK-NEXT: "use"(%idx2)
    %2 = pop.load %0 : !pop.pointer<index>
    "use"(%2) : (index) -> ()
    lit.try.yield
  // CHECK: }
  }
  // CHECK-NEXT: "use"(%idx3)
  pop.store %idx3, %0 : !pop.pointer<index>
  %2 = pop.load %0 : !pop.pointer<index>
  "use"(%2) : (index) -> ()
  kgen.return
}
