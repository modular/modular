// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @stack_allocation
kgen.func @stack_allocation(%cond: i1) {
  // CHECK-NEXT: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-NEXT: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32
  // CHECK-NEXT: llvm.intr.lifetime.start 64, %[[PTR0]]
  %0 = pop.stack_allocation 16 x !pop.simd<1, f32>
  // CHECK: hlcf.if
  hlcf.if %cond {
    // CHECK-NEXT: %[[C4:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK-NEXT: %[[PTR1:.*]] = llvm.alloca %[[C4]] x vector<4xf32>
    // CHECK-NEXT: llvm.intr.lifetime.start 64, %[[PTR1]]
    // CHECK-NEXT: llvm.intr.lifetime.end 64, %[[PTR1]]
    %1 = pop.stack_allocation 4 x !pop.simd<4, f32>
    // CHECK: }
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: llvm.intr.lifetime.end 64, %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_with_alignment
kgen.func @stack_allocation_with_alignment() {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32 {alignment = 8 : i64}
  // CHECK: llvm.intr.lifetime.start 64, %[[PTR0]]
  %0 = pop.stack_allocation 16 x !pop.simd<1, f32> align 8
  // CHECK-NEXT: llvm.intr.lifetime.end 64, %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_with_addressspace
kgen.func @stack_allocation_with_addressspace() {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x i32 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.intr.lifetime.start 64, %[[PTR0]]
  %0 = pop.stack_allocation 16 x !pop.simd<1, si32> address_space 5
  // CHECK-NEXT: llvm.intr.lifetime.end 64, %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_with_align_and_addressspace
kgen.func @stack_allocation_with_align_and_addressspace() {
  // CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[PTR0:.*]] = llvm.alloca %[[C16]] x f32 {alignment = 8 : i64} : (i64) -> !llvm.ptr<3>
  // CHECK: llvm.intr.lifetime.start 64, %[[PTR0]]
  %0 = pop.stack_allocation 16 x !pop.simd<1, f32> address_space 3 align 8
  // CHECK-NEXT: llvm.intr.lifetime.end 64, %[[PTR0]]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @stack_allocation_insertion
kgen.func @stack_allocation_insertion(%v: !pop.simd<1, si32>) {
  // CHECK: hlcf.loop
  hlcf.loop {
    // CHECK: llvm.alloca
    // CHECK: llvm.intr.lifetime.start
    %2 = pop.stack_allocation 1 x !pop.simd<1, si32>
    // CHECK: llvm.intr.lifetime.end
    // CHECK: hlcf.break
    hlcf.break
  }
  kgen.return
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {
  // CHECK-LABEL @allocate_64_bit
  kgen.func @allocate_64_bit() {
    // CHECK: lifetime.start 8, {{.*}}
    %0 = pop.stack_allocation 1 x index
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:32:32", simd_bit_width=128>} {
  // CHECK-LABEL @allocate_32_bit
  kgen.func @allocate_32_bit() {
    // CHECK: lifetime.start 4, {{.*}}
    %0 = pop.stack_allocation 1 x index
    kgen.return
  }
}
