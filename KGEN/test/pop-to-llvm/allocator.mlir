// RUN: kgen-opt %s -lower-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

// CHECK: declare noalias ptr @kgenAlignedAlloc(i64, i64 allocalign) [[ALLOC_ATTRS:#[0-9]+]]
// CHECK: declare void @kgenAlignedFree(ptr allocptr) [[FREE_ATTRS:#[0-9]+]]

// CHECK: attributes [[ALLOC_ATTRS]] = 
// CHECK-SAME: allockind("alloc,uninitialized,aligned")
// CHECK-SAME: allocsize(0)
// CHECK-SAME: "alloc-family"="kgen_aligned_allocator"

// CHECK: attributes [[FREE_ATTRS]] =
// CHECK-SAME: allockind("free")
// CHECK-SAME: "alloc-family"="kgen_aligned_allocator"

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @alloc_free() {
    %size = index.constant 1
    %align = index.constant 8
    %0 = pop.aligned_alloc %size, %align : <index>
    pop.aligned_free %0 : <index>
    kgen.return
  }
}
