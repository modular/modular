// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_size=8, simd_bit_width=128>} {

// CHECK-LABEL: @stack_allocation
kgen.func @stack_allocation(%cond: i1) {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[C4:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32
  // CHECK-DAG: %[[PTR1:.*]] = llvm.alloca %[[C4]] x vector<4xf32>
  // CHECK: llvm.intr.lifetime.start 64, %[[PTR0]]
  %0 = pop.stack_allocation 16 x !pop.simd<1, f32>
  // CHECK: scf.if
  scf.if %cond {
    // CHECK-NEXT: llvm.intr.lifetime.start 64, %[[PTR1]]
    // CHECK-NEXT: llvm.intr.lifetime.end 64, %[[PTR1]]
    %1 = pop.stack_allocation 4 x !pop.simd<4, f32>
    // CHECK: }
  }
  // CHECK-NEXT: llvm.intr.lifetime.end 64, %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_with_alignment
kgen.func @stack_allocation_with_alignment(%cond: i1) {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32 {alignment = 8 : i64}
  // CHECK: llvm.intr.lifetime.start 64, %[[PTR0]]
  %0 = pop.stack_allocation 16 x !pop.simd<1, f32> align 8
  // CHECK-NEXT: llvm.intr.lifetime.end 64, %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_insertion
kgen.func @stack_allocation_insertion(%v: !pop.simd<1, si32>, %lb: index, %ub: index, %step: index) {
  // CHECK: llvm.alloca
  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%sum = %v) -> !pop.simd<1, si32> {
    %0 = index.casts %i : index to i32
    %1 = pop.cast_from_builtin %0 : i32 to !pop.simd<1, si32>
    // CHECK: llvm.intr.lifetime.start
    %2 = pop.stack_allocation 1 x !pop.simd<1, si32>
    pop.store %sum, %2 : !pop.pointer<simd<1, si32>>
    %3 = pop.add %1, %sum : !pop.simd<1, si32>
    // CHECK: llvm.intr.lifetime.end
    // CHECK: scf.yield
    scf.yield %3 : !pop.simd<1, si32>
  }
  kgen.return
}

}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_size=8, simd_bit_width=128>} {
  // CHECK-LABEL @allocate_64_bit
  kgen.func @allocate_64_bit() {
    // CHECK: lifetime.start 8, {{.*}}
    %0 = pop.stack_allocation 1 x index
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_size=4, simd_bit_width=128>} {
  // CHECK-LABEL @allocate_32_bit
  kgen.func @allocate_32_bit() {
    // CHECK: lifetime.start 4, {{.*}}
    %0 = pop.stack_allocation 1 x index
    kgen.return
  }
}
