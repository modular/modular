// RUN: kgen-opt -pass-pipeline='kgen.func(lower-pop-to-llvm)' %s | FileCheck %s

// CHECK-LABEL: @stack_allocation
kgen.func @stack_allocation(%cond: i1) {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[C4:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32
  // CHECK-DAG: %[[PTR1:.*]] = llvm.alloca %[[C4]] x vector<4xf32>
  // CHECK: llvm.intr.lifetime.start 16, %[[PTR0]]
  %0 = pop.stack_allocation 16 : !meta.scalar<f32>
  // CHECK: scf.if
  scf.if %cond {
    // CHECK-NEXT: llvm.intr.lifetime.start 4, %[[PTR1]]
    // CHECK-NEXT: llvm.intr.lifetime.end 4, %[[PTR1]]
    %1 = pop.stack_allocation 4 : !meta.simd<4, f32>
    // CHECK: }
  }
  // CHECK-NEXT: llvm.intr.lifetime.end 16, %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_insertion
kgen.func @stack_allocation_insertion(%v: !meta.scalar<si32>, %lb: index, %ub: index, %step: index) {
  // CHECK: llvm.alloca
  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%sum = %v) -> !meta.scalar<si32> {
    %0 = index.casts %i : index to i32
    %1 = pop.type_raise %0 : i32 to !meta.scalar<si32>
    // CHECK: llvm.intr.lifetime.start
    %2 = pop.stack_allocation 1 : !meta.scalar<si32>
    pop.store %sum, %2 : !pop.pointer<!meta.scalar<si32>>
    %3 = pop.add %1, %sum : !meta.scalar<si32>
    // CHECK: llvm.intr.lifetime.end
    // CHECK: scf.yield
    scf.yield %3 : !meta.scalar<si32>
  }
  kgen.return
}
