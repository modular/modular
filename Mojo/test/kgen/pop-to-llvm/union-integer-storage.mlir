// RUN: kgen-opt %s -split-input-file -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm,canonicalize))' | FileCheck %s

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="generic", features="", data_layout="e-m:e-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// Large Optional payloads use i64 tail elements on 64-bit targets to reduce
// potential bytewise scalarization (#7391).
// CHECK-LABEL: @large_union
// CHECK-SAME: !llvm.struct<(i64, array<127 x i64>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i64, array<127 x i64>)>
kgen.func @large_union(%arg: !pop.union<i64, array<1024, i8>>) -> !pop.union<i64, array<1024, i8>> {
  hlcf.return %arg : !pop.union<i64, array<1024, i8>>
}

// Integer tail storage must not increase the union's alignment.
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

module attributes {M.target_info = #M.target<triple="i386-unknown-linux-gnu", arch="generic", features="", data_layout="e-m:e-p:32:32-i64:32-i128:128-n32-S128", simd_bit_width=128>} {

// A four-byte tail can use i32 storage on a 32-bit target.
// CHECK-LABEL: @i386_i32_tail
// CHECK-SAME: !llvm.struct<(i64, array<1 x i32>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i64, array<1 x i32>)>
kgen.func @i386_i32_tail(%arg: !pop.union<i64, array<12, i8>>) -> !pop.union<i64, array<12, i8>> {
  hlcf.return %arg : !pop.union<i64, array<12, i8>>
}

}

// -----

module attributes {M.target_info = #M.target<triple="riscv32-unknown-linux-gnu", arch="generic-rv32", features="", data_layout="e-m:e-p:32:32-i64:64-n32-S128", simd_bit_width=128>} {

// Large tails use i32 storage on RV32 without introducing i64 elements.
// CHECK-LABEL: @rv32_large_union
// CHECK-SAME: !llvm.struct<(i32, array<255 x i32>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i32, array<255 x i32>)>
kgen.func @rv32_large_union(%arg: !pop.union<i32, array<1024, i8>>) -> !pop.union<i32, array<1024, i8>> {
  hlcf.return %arg : !pop.union<i32, array<1024, i8>>
}

// Even when i64 has suitable alignment, the tail must use i32 elements.
// CHECK-LABEL: @rv32_aligned_tail
// CHECK-SAME: !llvm.struct<(f64, array<2 x i32>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(f64, array<2 x i32>)>
kgen.func @rv32_aligned_tail(%arg: !pop.union<f64, array<16, i8>>) -> !pop.union<f64, array<16, i8>> {
  hlcf.return %arg : !pop.union<f64, array<16, i8>>
}

// Integer tail storage must not increase alignment on 32-bit targets either.
// CHECK-LABEL: @rv32_byte_aligned_union
// CHECK-SAME: !llvm.struct<(i8, array<15 x i8>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i8, array<15 x i8>)>
kgen.func @rv32_byte_aligned_union(%arg: !pop.union<i8, array<16, i8>>) -> !pop.union<i8, array<16, i8>> {
  hlcf.return %arg : !pop.union<i8, array<16, i8>>
}

}

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="generic", features="", data_layout="e-m:e-i64:32-i128:128-n32:64-S128", simd_bit_width=128>} {

// A partial i64 tail must fall back to bytes on a 64-bit target.
// CHECK-LABEL: @partial_i64_tail
// CHECK-SAME: !llvm.struct<(i64, array<4 x i8>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i64, array<4 x i8>)>
kgen.func @partial_i64_tail(%arg: !pop.union<i64, array<12, i8>>) -> !pop.union<i64, array<12, i8>> {
  hlcf.return %arg : !pop.union<i64, array<12, i8>>
}

}

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="generic", features="", data_layout="e-m:e-i64:128-i128:128-n32:64-S128", simd_bit_width=128>} {

// An i64 whose allocation includes padding cannot represent every payload byte.
// CHECK-LABEL: @padded_i64_type
// CHECK-SAME: !llvm.struct<(i128, array<16 x i8>)>
// CHECK: llvm.return %{{.*}} : !llvm.struct<(i128, array<16 x i8>)>
kgen.func @padded_i64_type(%arg: !pop.union<i128, array<32, i8>>) -> !pop.union<i128, array<32, i8>> {
  hlcf.return %arg : !pop.union<i128, array<32, i8>>
}

}
