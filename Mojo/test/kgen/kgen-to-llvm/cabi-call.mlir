// RUN: kgen-opt -lower-kgen-to-llvm -lower-runtime-closures -split-input-file %s | FileCheck %s
//
// Tests for kgen.call / kgen.call_indirect → llvm.call lowering for abi("C")
// callees.
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/LowerKGENToLLVM.cpp  — ConvertKGENCall
//   Mojo/lib/KGENToLLVM/LowerKGENClosuresToLLVM.cpp  — CallIndirectOpConversion
//   Mojo/lib/KGENToLLVM/CABICallHelpers.cpp  — CABICallHelper
//
// Focus: interaction between TailKind and the sret return convention (T1-T4)
// and by-memory arguments (T4a-T4b), and the extension of narrow integers
// (T5-T6).
//
// On aarch64 AAPCS:
//   {i64,i64,i64} (24 bytes) → sret (hidden pointer in x8)
//   {i32}          (4 bytes)  → integer coercion → i32 register return
//
// A kgen.call marked tail to an abi("C") callee that returns via sret must NOT
// produce a `tail call` in the lowered LLVM IR.  The sret pointer is a local
// alloca on the caller's frame; a tail call would free that frame before the
// callee writes through the pointer, corrupting the return value (MOCO-3841).

//===----------------------------------------------------------------------===//
// T1: sret return — tail call must be suppressed.
//
// {i64,i64,i64} (24 bytes) is sret on aarch64.  A kgen.call with TailKind::Tail
// to such a callee must lower to a regular llvm.call (no tail keyword), because
// the sret alloca lives in the caller's frame.
//===----------------------------------------------------------------------===//

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

kgen.func @callee_sret(%s: !llvm.struct<(i64, i64, i64)>) cabi -> !llvm.struct<(i64, i64, i64)> {
  hlcf.return %s : !llvm.struct<(i64, i64, i64)>
}

// CHECK-LABEL: llvm.func internal @caller_sret
kgen.func @caller_sret(%s: !llvm.struct<(i64, i64, i64)>) -> !llvm.struct<(i64, i64, i64)> {
  // Regular call (no tail): the sret alloca is on the caller's frame.
  // CHECK-NOT: llvm.call tail @callee_sret
  // CHECK:     llvm.call @callee_sret
  %r = kgen.call tail @callee_sret(%s) : (!llvm.struct<(i64, i64, i64)>) -> !llvm.struct<(i64, i64, i64)>
  hlcf.return %r : !llvm.struct<(i64, i64, i64)>
}

} // module

//===----------------------------------------------------------------------===//
// T2: register return — tail call must be preserved.
//
// {i32} (4 bytes) is returned in a register on aarch64 (coerced to i32).
// No sret alloca is created, so TailKind::Tail must flow through to the
// lowered llvm.call unchanged.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

kgen.func @callee_reg(%s: !llvm.struct<(i32)>) cabi -> !llvm.struct<(i32)> {
  hlcf.return %s : !llvm.struct<(i32)>
}

// CHECK-LABEL: llvm.func internal @caller_reg
kgen.func @caller_reg(%s: !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
  // Tail call preserved: register return has no sret alloca.
  // CHECK: llvm.call tail @callee_reg
  %r = kgen.call tail @callee_reg(%s) : (!llvm.struct<(i32)>) -> !llvm.struct<(i32)>
  hlcf.return %r : !llvm.struct<(i32)>
}

} // module

//===----------------------------------------------------------------------===//
// T3: indirect sret return — tail call must be suppressed.
//
// This mirrors T1 for a bare abi("C") function pointer. The indirect call path
// builds the same sret alloca, so it must also drop TailKind::Tail.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @caller_indirect_sret
kgen.func @caller_indirect_sret(
    %callee: !kgen.generator<(!llvm.struct<(i64, i64, i64)>) cabi -> !llvm.struct<(i64, i64, i64)>>,
    %s: !llvm.struct<(i64, i64, i64)>
) -> !llvm.struct<(i64, i64, i64)> {
  // Regular call (no tail): the sret alloca is on the caller's frame.
  // CHECK-NOT: llvm.call tail
  // CHECK:     llvm.call %arg0
  %r = kgen.call_indirect tail %callee(%s) : (!llvm.struct<(i64, i64, i64)>) cabi -> !llvm.struct<(i64, i64, i64)>
  hlcf.return %r : !llvm.struct<(i64, i64, i64)>
}

} // module

//===----------------------------------------------------------------------===//
// T4: indirect register return — tail call must be preserved.
//
// This mirrors T2 for a bare abi("C") function pointer. No sret alloca is
// created, so TailKind::Tail must flow through to the lowered llvm.call.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @caller_indirect_reg
kgen.func @caller_indirect_reg(
    %callee: !kgen.generator<(!llvm.struct<(i32)>) cabi -> !llvm.struct<(i32)>>,
    %s: !llvm.struct<(i32)>
) -> !llvm.struct<(i32)> {
  // Tail call preserved: register return has no sret alloca.
  // CHECK: llvm.call tail %arg0
  %r = kgen.call_indirect tail %callee(%s) : (!llvm.struct<(i32)>) cabi -> !llvm.struct<(i32)>
  hlcf.return %r : !llvm.struct<(i32)>
}

} // module

//===----------------------------------------------------------------------===//
// T4a: by-memory argument passed as a plain pointer — tail call suppressed.
//
// {i64,i64,i64} (24 bytes) is passed on aarch64 as a pointer to a copy in the
// caller's frame, with no `byval`, even though the result comes back in a
// register. The callee reads that copy, so a tail call would free it first.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

kgen.func @callee_mem_arg(%s: !llvm.struct<(i64, i64, i64)>) cabi -> i64 {
  %r = llvm.extractvalue %s[0] : !llvm.struct<(i64, i64, i64)>
  hlcf.return %r : i64
}

// The pointer the callee receives is this `alloca`, and it carries no `byval`,
// so the copy belongs to this frame.
// CHECK-LABEL: llvm.func internal @caller_mem_arg
// CHECK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(i64, i64, i64)>
// CHECK:         llvm.store %arg0, %[[SLOT]]
// CHECK-NOT:     llvm.call tail
// CHECK:         llvm.call @callee_mem_arg(%[[SLOT]]) : (!llvm.ptr) -> i64
kgen.func @caller_mem_arg(%s: !llvm.struct<(i64, i64, i64)>) -> i64 {
  %r = kgen.call tail @callee_mem_arg(%s) : (!llvm.struct<(i64, i64, i64)>) -> i64
  hlcf.return %r : i64
}

// CHECK-LABEL: llvm.func internal @caller_indirect_mem_arg
// CHECK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(i64, i64, i64)>
// CHECK-NOT:     llvm.call tail
// CHECK:         llvm.call %arg0(%[[SLOT]]) : !llvm.ptr, (!llvm.ptr) -> i64
kgen.func @caller_indirect_mem_arg(
    %callee: !kgen.generator<(!llvm.struct<(i64, i64, i64)>) cabi -> i64>,
    %s: !llvm.struct<(i64, i64, i64)>
) -> i64 {
  %r = kgen.call_indirect tail %callee(%s) : (!llvm.struct<(i64, i64, i64)>) cabi -> i64
  hlcf.return %r : i64
}

} // module

//===----------------------------------------------------------------------===//
// T4b: the same argument on x86_64 SystemV, where it is `byval` — tail call
// kept. `byval` makes the call sequence copy the pointee, so the callee reads
// that copy rather than the caller's frame.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {

kgen.func @callee_byval_x86(%s: !llvm.struct<(i64, i64, i64)>) cabi -> i64 {
  %r = llvm.extractvalue %s[0] : !llvm.struct<(i64, i64, i64)>
  hlcf.return %r : i64
}

// Same `alloca`-and-store shape as T4a; the `llvm.byval` on the call argument
// is the whole difference, so the CHECK pins it.
// CHECK-LABEL: llvm.func internal @caller_byval_x86
// CHECK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(i64, i64, i64)>
// CHECK:         llvm.store %arg0, %[[SLOT]]
// CHECK:         llvm.call tail @callee_byval_x86(%[[SLOT]]) : (!llvm.ptr {llvm.byval = !llvm.struct<(i64, i64, i64)>}) -> i64
kgen.func @caller_byval_x86(%s: !llvm.struct<(i64, i64, i64)>) -> i64 {
  %r = kgen.call tail @callee_byval_x86(%s) : (!llvm.struct<(i64, i64, i64)>) -> i64
  hlcf.return %r : i64
}

} // module

//===----------------------------------------------------------------------===//
// T5: narrow integer extension on Darwin arm64 (MOCO-5051).
//
// Apple's arm64 ABI requires the producer to extend a narrow integer, and
// clang relies on it at the consumer. The sign comes from the KGEN dtype;
// Bool is zero-extended.
//
// Only values Mojo produces are marked: a definition marks its result and a
// call marks its arguments. A mark on a definition parameter or a call result
// would let LLVM assume the C side extended it, but a C struct flattened to a
// bare Mojo integer arrives with unspecified upper bits.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// The definition extends its result only.
// CHECK-LABEL: llvm.func internal @callee_narrow
// CHECK-SAME:    (%{{.*}}: i8 {llvm.noundef}, %{{.*}}: i16 {llvm.noundef}, %{{.*}}: i1 {llvm.noundef}, %{{.*}}: i32 {llvm.noundef})
// CHECK-SAME:    -> (i8 {llvm.signext})
kgen.func @callee_narrow(%u: !kgen.scalar<ui8>, %s: !kgen.scalar<si16>, %b: !kgen.scalar<bool>, %w: !kgen.scalar<si32>) cabi -> !kgen.scalar<si8> {
  %r = pop.cast %u : !kgen.scalar<ui8> to !kgen.scalar<si8>
  hlcf.return %r : !kgen.scalar<si8>
}

// The call extends its arguments only; i32 needs no extension.
// CHECK-LABEL: llvm.func internal @caller_narrow
// CHECK:         llvm.call @callee_narrow(%{{.*}}) : (i8 {llvm.zeroext}, i16 {llvm.signext}, i1 {llvm.zeroext}, i32) -> i8
kgen.func @caller_narrow(%u: !kgen.scalar<ui8>, %s: !kgen.scalar<si16>, %b: !kgen.scalar<bool>, %w: !kgen.scalar<si32>) -> !kgen.scalar<si8> {
  %r = kgen.call @callee_narrow(%u, %s, %b, %w) : (!kgen.scalar<ui8>, !kgen.scalar<si16>, !kgen.scalar<bool>, !kgen.scalar<si32>) -> !kgen.scalar<si8>
  hlcf.return %r : !kgen.scalar<si8>
}

// CHECK-LABEL: llvm.func internal @caller_indirect_narrow
// CHECK:         llvm.call %arg0(%{{.*}}) : !llvm.ptr, (i8 {llvm.signext}, i16 {llvm.zeroext}) -> i8
kgen.func @caller_indirect_narrow(
    %callee: !kgen.generator<(!kgen.scalar<si8>, !kgen.scalar<ui16>) cabi -> !kgen.scalar<ui8>>,
    %s: !kgen.scalar<si8>, %u: !kgen.scalar<ui16>
) -> !kgen.scalar<ui8> {
  %r = kgen.call_indirect %callee(%s, %u) : (!kgen.scalar<si8>, !kgen.scalar<ui16>) cabi -> !kgen.scalar<ui8>
  hlcf.return %r : !kgen.scalar<ui8>
}

// A struct of one narrow integer is coerced to an integer of its size, which
// clang does not extend, so neither the definition nor the call marks it,
// whatever the sign of the field.
// CHECK-LABEL: llvm.func internal @callee_single_field
// CHECK-SAME:    (%{{.*}}: i8 {llvm.noundef}, %{{.*}}: i16 {llvm.noundef}, %{{.*}}: i8 {llvm.noundef}) -> i8 attributes
kgen.func @callee_single_field(%s: !kgen.struct<(scalar<si8>)>, %u: !kgen.struct<(scalar<ui16>)>, %b: !kgen.struct<(scalar<bool>)>) cabi -> !kgen.struct<(scalar<si8>)> {
  hlcf.return %s : !kgen.struct<(scalar<si8>)>
}

// CHECK-LABEL: llvm.func internal @caller_single_field
// CHECK:         llvm.call @callee_single_field(%{{.*}}) : (i8, i16, i8) -> i8
kgen.func @caller_single_field(%s: !kgen.struct<(scalar<si8>)>, %u: !kgen.struct<(scalar<ui16>)>, %b: !kgen.struct<(scalar<bool>)>) -> !kgen.struct<(scalar<si8>)> {
  %r = kgen.call @callee_single_field(%s, %u, %b) : (!kgen.struct<(scalar<si8>)>, !kgen.struct<(scalar<ui16>)>, !kgen.struct<(scalar<bool>)>) -> !kgen.struct<(scalar<si8>)>
  hlcf.return %r : !kgen.struct<(scalar<si8>)>
}

} // module

//===----------------------------------------------------------------------===//
// T6: no narrow integer extension on Linux arm64.
//
// AAPCS64 leaves the upper bits unspecified and clang emits no extension, so
// the T5 signature gets no zeroext/signext.
//===----------------------------------------------------------------------===//

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @callee_narrow_linux
// CHECK-SAME:    -> i8 attributes
kgen.func @callee_narrow_linux(%u: !kgen.scalar<ui8>, %s: !kgen.scalar<si16>) cabi -> !kgen.scalar<si8> {
  %r = pop.cast %u : !kgen.scalar<ui8> to !kgen.scalar<si8>
  hlcf.return %r : !kgen.scalar<si8>
}

// CHECK-LABEL: llvm.func internal @caller_narrow_linux
// CHECK:         llvm.call @callee_narrow_linux(%{{.*}}) : (i8, i16) -> i8
kgen.func @caller_narrow_linux(%u: !kgen.scalar<ui8>, %s: !kgen.scalar<si16>) -> !kgen.scalar<si8> {
  %r = kgen.call @callee_narrow_linux(%u, %s) : (!kgen.scalar<ui8>, !kgen.scalar<si16>) -> !kgen.scalar<si8>
  hlcf.return %r : !kgen.scalar<si8>
}

} // module
