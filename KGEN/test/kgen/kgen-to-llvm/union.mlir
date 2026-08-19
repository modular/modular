// RUN: kgen-opt -lower-kgen-to-llvm %s | FileCheck %s

// Test lowering of `!pop.union` to LLVM.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @union_bool_simd_with_empty_member
// CHECK-SAME: !llvm.struct<(i16)>
kgen.func @union_bool_simd_with_empty_member(%arg0: !pop.union<!kgen.struct<()>, !kgen.simd<2, bool>>) {
  kgen.return
}

// CHECK-LABEL: @union_bool_simd
// CHECK-SAME: !llvm.struct<(i16)>
kgen.func @union_bool_simd(%arg0: !pop.union<!kgen.simd<2, bool>>) {
  kgen.return
}

// A wider member still dominates the layout.
// CHECK-LABEL: @union_bool_simd_with_wider_member
// CHECK-SAME: !llvm.struct<(i64)>
kgen.func @union_bool_simd_with_wider_member(%arg0: !pop.union<index, !kgen.simd<2, bool>>) {
  kgen.return
}

}
