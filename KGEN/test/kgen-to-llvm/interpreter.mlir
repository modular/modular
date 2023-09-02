// RUN: kgen-opt %s -lower-kgen-to-llvm | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @heap
kgen.func @heap() -> !kgen.pointer<i16> {
  // CHECK: %[[ALLOC:.*]] = pop.aligned_alloc %idx32, %idx2 : <i8>
  // CHECK: %[[ALLOC_LLVM:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]] : !kgen.pointer<i8> to !llvm.ptr<i8>
  // CHECK: %[[BASE:.*]] = llvm.bitcast %[[ALLOC_LLVM]] : !llvm.ptr<i8> to !llvm.ptr
  // CHECK: %[[P0:.*]] = llvm.getelementptr inbounds %[[BASE]][0]
  // CHECK: %[[CST_EF:.*]] = llvm.mlir.constant(-17 :
  // CHECK: llvm.store %[[CST_EF]], %[[P0]] {alignment = 32 :
  // CHECK: %[[P1:.*]] = llvm.getelementptr inbounds %[[BASE]][1]
  // CHECK: %[[CST_BE:.*]] = llvm.mlir.constant(-66 :
  // CHECK: llvm.store %[[CST_BE]], %[[P1]] {alignment = 32 :
  // CHECK: %[[RESULT:.*]] = llvm.getelementptr inbounds %[[BASE]][0]
  // CHECK: %[[RESULT_TYPED:.*]] = llvm.bitcast %[[RESULT]]
  %0 = kgen.param.materialize: !kgen.pointer<i16> = <#interp.memref<[(mem_heap, heap, [])], 0, 0>>
  // CHECK: llvm.return %[[RESULT_TYPED]] : !llvm.ptr<i16>
  kgen.return %0 : !kgen.pointer<i16>
}

// CHECK-LABEL: llvm.func internal @stack
kgen.func @stack() {
  // CHECK: %[[ALLOC:.*]] = pop.stack_allocation 2 x i8 align 32
  // CHECK-NEXT: builtin.unrealized_conversion_cast %[[ALLOC]]
  %0 = kgen.param.materialize: !kgen.pointer<i16> = <#interp.memref<[(mem_stack, stack, [])], 0, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @stack_shared
kgen.func @stack_shared() {
  // CHECK: %[[ALLOC:.*]] = pop.stack_allocation 2 x i8 align 32
  %0 = kgen.param.materialize: !kgen.pointer<i16> = <#interp.memref<[(mem_stack, stack, [])], 0, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @global
kgen.func @global() -> !kgen.pointer<i8> {
  // CHECK: %[[BASE:.*]] = llvm.mlir.addressof @mem_global : !llvm.ptr<array<4 x i8>>
  // CHECK: %[[BASE_OPAQUE:.*]] = llvm.bitcast %[[BASE]]
  // CHECK: %[[RESULT:.*]] = llvm.getelementptr inbounds %[[BASE_OPAQUE]][2]
  // CHECK: %[[RESULT_TYPED:.*]] = llvm.bitcast %[[RESULT]]
  %0 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<[(mem_global, const_global, [])], 0, 2>>
  // CHECK: llvm.return %[[RESULT_TYPED]]
  kgen.return %0 : !kgen.pointer<i8>
}

// CHECK-LABEL: llvm.func internal @pointer_to_pointer
kgen.func @pointer_to_pointer() {
  // CHECK: %[[ALLOC1:.*]] = pop.stack_allocation 9 x i8 align 16
  // CHECK: %[[PTR1_TYPED:.*]] = builtin.unrealized_conversion_cast %[[ALLOC1]]
  // CHECK: %[[PTR1:.*]] = llvm.bitcast %[[PTR1_TYPED]]

  // CHECK: %[[ALLOC2:.*]] = pop.aligned_alloc %idx16, %idx2
  // CHECK: %[[PTR2_TYPED:.*]] = builtin.unrealized_conversion_cast %[[ALLOC2]]
  // CHECK: %[[PTR2:.*]] = llvm.bitcast %[[PTR2_TYPED]]

  // CHECK: %[[PTR_REGION:.*]] = llvm.getelementptr inbounds %[[PTR1]][0]
  // CHECK: %[[PTEE:.*]] = llvm.getelementptr inbounds %[[PTR2]][0]
  // CHECK: llvm.store %[[PTEE]], %[[PTR_REGION]]

  // CHECK: %[[TAIL:.*]] = llvm.getelementptr inbounds %[[PTR1]][8]
  // CHECK: %[[C8:.*]] = llvm.mlir.constant(8 :
  // CHECK: llvm.store %[[C8]], %[[TAIL]]
  %0 = kgen.param.materialize: !kgen.pointer<pointer<i16>> = <#interp.memref<[
    (foo, stack, [(0, 1, 0)]),
    (bar, heap, [])
  ], 0, 0>>
  kgen.return
}

// CHECK-LABEL: llvm.func internal @string
kgen.func @string() {
  // CHECK: llvm.mlir.addressof @mem_string :
  %0 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<[(mem_string, const_global, [])], 0, 0>>
  // COM: Ensure `const_global` handle gets deduplicated.
  // CHECK: llvm.mlir.addressof @mem_string :
  %1 = kgen.param.materialize: !kgen.pointer<i8> = <#interp.memref<[(mem_string, const_global, []), (mem_stack, stack, [])], 0, 0>>
  kgen.return
}

// CHECK: llvm.mlir.global internal constant @mem_global(#M.dense_array<1, 2, 3, 4> : !M.array<4xi8>) {addr_space = 0 : i32, alignment = 32 : i64} : !llvm.array<4 x i8>
// CHECK: llvm.mlir.global internal constant @mem_string("hello world")

}

{-#
  dialect_resources: {
    interp: {
      mem_stack: "0x20000000ADDE",
      mem_heap: "0x20000000EFBE",
      mem_global: "0x2000000001020304",
      mem_string: "hello world",
      foo: "0x10000000000000000000000008",
      bar: "0x100000000000"
    }
  }
#-}
