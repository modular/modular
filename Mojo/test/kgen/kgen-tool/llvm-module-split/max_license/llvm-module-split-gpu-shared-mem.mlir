// RUN: kgen --emit=llvm %s -o %t
// RUN: cat %t | FileCheck %s --check-prefix=CHECK-EMIT-LLVM
// RUN: llvm-module-split %t | FileCheck %s --check-prefix=CHECK-SPLIT

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_86", features = "+ptx81,+sm_86", data_layout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128, index_bit_width = 64, tune_cpu = "sm_86">, llvm.data_layout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", llvm.target_triple = "nvptx64-nvidia-cuda"} {

//CHECK-EMIT-LLVM: f._global_alloc_0._gpu_shared_mem = internal addrspace(3) global
//CHECK-EMIT-LLVM: extern_ptr_syml = external dso_local addrspace(3) global
kgen.func @f() no_inline {
   %gv = pop.global_alloc "_global_alloc" 1 x scalar<f32> address_space 3  align 4 {memoryType = #pop.global_alloc_addr_space<gpu_shared>}
   %ptr = pop.extern_ptr_symbol "extern_ptr_syml" alignment<4> : <array<0, scalar<f32>>, 3>
   %0 = pop.pointer.bitcast %ptr : !kgen.pointer<array<0, scalar<f32>>, 3> to !kgen.pointer<scalar<f32>, 3>
   %1 = pop.load %0 : !kgen.pointer<scalar<f32>, 3>
   pop.store %1, %gv : !kgen.pointer<scalar<f32>, 3>
   hlcf.return
}

kgen.func export @g0() {
    kgen.call @f() : () -> ()
    hlcf.return
}

kgen.func export @g1() {
    kgen.call @f() : () -> ()
    hlcf.return
}

}

// COM: check splitting global_alloc in GPU shared memory (addr_space 3)
// is not considered as splitting anchor, and can be split
// into separate modules.

// CHECK-SPLIT: [LLVM Module Split: submodule 0]
// CHECK-SPLIT: @f._global_alloc_0._gpu_shared_mem = internal addrspace(3)
// CHECK-SPLIT: extern_ptr_syml = external dso_local addrspace(3) global
// CHECK-SPLIT: g0()

// CHECK-SPLIT: [LLVM Module Split: submodule 1]
// CHECK-SPLIT: @f._global_alloc_0._gpu_shared_mem = internal addrspace(3)
// CHECK-SPLIT: extern_ptr_syml = external dso_local addrspace(3) global
// CHECK-SPLIT: g1()
