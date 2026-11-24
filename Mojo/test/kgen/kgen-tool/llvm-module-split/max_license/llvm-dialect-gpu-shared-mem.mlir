// RUN: llvm-module-split %s | FileCheck %s

// A global in GPU shared memory is local to a block, so each kernel's slice
// gets its own copy. The global neither roots a slice nor ties its users.

// CHECK-LABEL: [LLVM Module Split: submodule 0]
// CHECK-DAG: @f._global_alloc_0._gpu_shared_mem = internal addrspace(3) global
// CHECK-DAG: @extern_ptr_syml = external addrspace(3) global
// CHECK-DAG: define void @g0()
// CHECK-LABEL: [LLVM Module Split: submodule 1]
// CHECK-DAG: @f._global_alloc_0._gpu_shared_mem = internal addrspace(3) global
// CHECK-DAG: @extern_ptr_syml = external addrspace(3) global
// CHECK-DAG: define void @g1()

module attributes {llvm.data_layout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", llvm.target_triple = "nvptx64-nvidia-cuda"} {
  llvm.mlir.global internal @f._global_alloc_0._gpu_shared_mem() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<1 x f32> {
    %0 = llvm.mlir.undef : !llvm.array<1 x f32>
    llvm.return %0 : !llvm.array<1 x f32>
  }

  llvm.mlir.global external @extern_ptr_syml() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<0 x f32>

  llvm.func internal @f() {
    %0 = llvm.mlir.addressof @f._global_alloc_0._gpu_shared_mem : !llvm.ptr<3>
    %1 = llvm.mlir.addressof @extern_ptr_syml : !llvm.ptr<3>
    %2 = llvm.load %1 : !llvm.ptr<3> -> f32
    llvm.store %2, %0 : f32, !llvm.ptr<3>
    llvm.return
  }

  llvm.func @g0() {
    llvm.call @f() : () -> ()
    llvm.return
  }

  llvm.func @g1() {
    llvm.call @f() : () -> ()
    llvm.return
  }
}
