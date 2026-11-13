// RUN: kgen-opt -lower-kgen-to-llvm -split-input-file %s | FileCheck %s
//
// Tests for the parameter attributes of abi("C") definitions.
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/LowerKGENToLLVMRewriteCABIFns.cpp
//   Mojo/lib/KGENToLLVM/CABILowering.cpp - SignatureClassification
//
// The attributes of each argument move to the LLVM parameter that the
// argument becomes. An sret pointer changes
// this position. A struct that is passed by value does not get noundef,
// because some of its bytes are not defined. These are its padding bytes
// and the bytes that a coercion adds after the end of the struct.

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// The ABI coerces a 3-byte struct to i64.
// CHECK-LABEL: llvm.func internal @callee_s3
// CHECK-SAME:    (%{{[a-z0-9]+}}: i64) -> i8
kgen.func @callee_s3(%s: !llvm.struct<(i8, i8, i8)>) cabi -> i8 {
  %r = llvm.extractvalue %s[0] : !llvm.struct<(i8, i8, i8)>
  hlcf.return %r : i8
}

// The ABI coerces an 8-byte struct to i64. The sizes are equal, but the
// parameter does not get noundef.
// CHECK-LABEL: llvm.func internal @callee_s8
// CHECK-SAME:    (%{{[a-z0-9]+}}: i64) -> i32
kgen.func @callee_s8(%s: !llvm.struct<(i32, i32)>) cabi -> i32 {
  %r = llvm.extractvalue %s[0] : !llvm.struct<(i32, i32)>
  hlcf.return %r : i32
}

// Bytes 4-7 of the struct are padding. They are in the first of the two
// registers. The scalar arguments after the struct keep noundef.
// CHECK-LABEL: llvm.func internal @callee_pad
// CHECK-SAME:    (%{{[a-z0-9]+}}: !llvm.array<2 x i64>, %{{[a-z0-9]+}}: i8 {llvm.noundef}, %{{[a-z0-9]+}}: i64 {llvm.noundef}) -> i64
kgen.func @callee_pad(%p: !llvm.struct<(i32, i64)>, %x: i8, %y: i64) cabi -> i64 {
  hlcf.return %y : i64
}

// Bytes 13-15 of the struct are tail padding. They are in the second
// register.
// CHECK-LABEL: llvm.func internal @callee_s13
// CHECK-SAME:    (%{{[a-z0-9]+}}: !llvm.array<2 x i64>) -> i64
kgen.func @callee_s13(%s: !llvm.struct<(i64, i32, i8)>) cabi -> i64 {
  %r = llvm.extractvalue %s[0] : !llvm.struct<(i64, i32, i8)>
  hlcf.return %r : i64
}

// The sret pointer moves each argument to the next parameter.
// CHECK-LABEL: llvm.func internal @callee_sret
// CHECK-SAME:    (%{{[a-z0-9]+}}: !llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, %{{[a-z0-9]+}}: i64, %{{[a-z0-9]+}}: i64 {llvm.noundef})
kgen.func @callee_sret(%s: !llvm.struct<(i8, i8, i8)>, %y: i64) cabi -> !llvm.struct<(i64, i64, i64)> {
  %u = llvm.mlir.undef : !llvm.struct<(i64, i64, i64)>
  %r = llvm.insertvalue %y, %u[0] : !llvm.struct<(i64, i64, i64)>
  hlcf.return %r : !llvm.struct<(i64, i64, i64)>
}

// An all-float struct becomes an array of its fields. It does not get
// noundef.
// CHECK-LABEL: llvm.func internal @callee_hfa
// CHECK-SAME:    (%{{[a-z0-9]+}}: !llvm.array<2 x f32>, %{{[a-z0-9]+}}: i64 {llvm.noundef}) -> i64
kgen.func @callee_hfa(%s: !llvm.struct<(f32, f32)>, %y: i64) cabi -> i64 {
  hlcf.return %y : i64
}

// The ABI passes this struct by pointer. The pointer value is fully
// defined, so it keeps noundef.
// CHECK-LABEL: llvm.func internal @callee_indirect
// CHECK-SAME:    (%{{[a-z0-9]+}}: !llvm.ptr {llvm.noundef}, %{{[a-z0-9]+}}: i64 {llvm.noundef}) -> i64
kgen.func @callee_indirect(%s: !llvm.struct<(i64, i64, i64)>, %y: i64) cabi -> i64 {
  hlcf.return %y : i64
}

} // module
