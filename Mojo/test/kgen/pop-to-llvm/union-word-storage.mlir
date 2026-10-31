// RUN: kgen-opt %s -split-input-file -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm,canonicalize))' | FileCheck %s

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="generic", features="", data_layout="e-m:e-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// Large Optional payloads must not become thousands of individual byte
// accesses during LLVM SROA and SelectionDAG store merging (#7391).
// CHECK-LABEL: @large_union
// CHECK-SAME: !llvm.struct<(i64, array<127 x i64>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i64, array<127 x i64>)>
kgen.func @large_union(%arg: !pop.union<i64, array<1024, i8>>) -> !pop.union<i64, array<1024, i8>> {
  hlcf.return %arg : !pop.union<i64, array<1024, i8>>
}

// Word-sized storage must not increase the union's alignment.
// CHECK-LABEL: @unaligned_tail
// CHECK-SAME: !llvm.struct<(i32, array<8 x i8>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i32, array<8 x i8>)>
kgen.func @unaligned_tail(%arg: !pop.union<i32, array<12, i8>>) -> !pop.union<i32, array<12, i8>> {
  hlcf.return %arg : !pop.union<i32, array<12, i8>>
}

// CHECK-LABEL: @byte_aligned_union
// CHECK-SAME: !llvm.struct<(i8, array<15 x i8>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i8, array<15 x i8>)>
kgen.func @byte_aligned_union(%arg: !pop.union<i8, array<16, i8>>) -> !pop.union<i8, array<16, i8>> {
  hlcf.return %arg : !pop.union<i8, array<16, i8>>
}

}

// -----

module attributes {M.target_info = #M.target<triple="i386-unknown-linux-gnu", arch="generic", features="", data_layout="e-m:e-i64:32-i128:128-n32-S128", simd_bit_width=128>} {

// Even when i64 has suitable alignment, a partial word must not be lost.
// CHECK-LABEL: @partial_word_tail
// CHECK-SAME: !llvm.struct<(i64, array<4 x i8>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i64, array<4 x i8>)>
kgen.func @partial_word_tail(%arg: !pop.union<i64, array<12, i8>>) -> !pop.union<i64, array<12, i8>> {
  hlcf.return %arg : !pop.union<i64, array<12, i8>>
}

}

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="generic", features="", data_layout="e-m:e-i64:128-i128:128-n32:64-S128", simd_bit_width=128>} {

// An i64 whose allocation includes padding cannot represent every payload byte.
// CHECK-LABEL: @padded_word_type
// CHECK-SAME: !llvm.struct<(i128, array<16 x i8>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i128, array<16 x i8>)>
kgen.func @padded_word_type(%arg: !pop.union<i128, array<32, i8>>) -> !pop.union<i128, array<32, i8>> {
  hlcf.return %arg : !pop.union<i128, array<32, i8>>
}

}
