// RUN: kgen-opt -lower-kgen-to-llvm -lower-global-pop-to-llvm -split-input-file %s | FileCheck %s

// Unit tests for C ABI lowering on ARM64 (AAPCS — Procedure Call Standard
// for the Arm 64-bit Architecture).
//
// PURPOSE: These are UNIT tests that lock the current behavior of the lowering
// transform. They exist to catch regressions during refactoring. They may need
// updating if we discover ABI bugs or add new features. The system tests are
// the authoritative check for correct end-to-end C ABI behavior:
//   KGEN/test/mojo-integration/extern-c-abi/
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/LowerPOPToLLVM.cpp  — ConvertPOPExternalCall
//   Mojo/lib/KGENToLLVM/CABIAAPCS.cpp        — AAPCSABIInfo classifier
//   Mojo/lib/KGENToLLVM/CABILowering.cpp     — shared utilities
//
// RELATED TEST FILES:
//   extern-c-abi.mlir          — platform-independent tests (DefaultCABIInfo,
//                                empty triple): struct not field-exploded,
//                                multi-value return, identity pass-through
//   extern-c-abi-x86-64.mlir   — x86-64 System V AMD64 tests
//
// INPUT FORMAT NOTE: Each test is a kgen.func whose operands and results of
// pop.external_call are KGEN types (!kgen.struct, !kgen.scalar), as the Mojo
// compiler emits them. The C ABI classifier reads these KGEN types. The pass
// -lower-kgen-to-llvm changes each kgen.func to an internal llvm.func, and
// then -lower-global-pop-to-llvm lowers the pop.external_call ops in it.
//
// KEY DIFFERENCES FROM x86-64:
//   1. HFA (Homogeneous Float Aggregate): a struct of 1-4 members of one
//      float type goes in SIMD registers (V0-V3). It is coerced to an array
//      of that type, because the backend keeps only an array argument in
//      consecutive registers.
//   2. Two-register structs always use IntegerPair (i64, iN) regardless of
//      field types. There is no SSEPair or Mixed class on ARM64.
//   3. Large struct args (>16 bytes) are passed indirectly WITHOUT byval.
//      x86-64 sets byval; ARM64 does not.
//   4. Variadic HFA args — Darwin vs Linux split:
//      Darwin (macOS/iOS): not classified as an HFA; struct coerced to
//        integer so va_arg reads from the GP save area (flat va_list).
//      Linux AAPCS64: classified as an HFA, as a fixed argument is, so LLVM
//        places it in SIMD registers and va_arg reads from the VR save area.
//   5. Variadic float args < 64 bits — Darwin vs Linux split:
//      Darwin: bitcast to integer to prevent LLVM float→double promotion
//        (Darwin va_list reads from GP only, so bits must be in GPRs).
//      Linux: no bitcast; LLVM places floats in SIMD registers and va_arg
//        reads from the VR save area correctly.
//
// TEST TABLE:
// Group A — Small struct args (1–8 bytes): Integer coercion (same as x86-64)
//   A1  arg_1byte_int_struct      {i8}       → i8
//   A2  arg_2byte_int_struct      {i8,i8}    → i16
//   A3  arg_3byte_int_struct      {i8,i8,i8} → i32
//   A4  arg_8byte_int_struct      {i32,i32}  → i64
//
// Group F - HFA (Homogeneous Float Aggregate): array of the base type
//   F1  arg_hfa_2xf32             {f32,f32}             -> [2 x f32] (SIMD)
//   F2  arg_hfa_4xf32             {f32,f32,f32,f32}     -> [4 x f32] (max HFA)
//   F3  arg_hfa_2xf64             {f64,f64}             -> [2 x f64] (SIMD)
//   F4  arg_not_hfa_5xf32         {f32,f32,f32,f32,f32} → indirect (>16 AND >4 fields)
//   F5  arg_not_hfa_mixed         {f32,i32}             → integer coercion (not all-float)
//   F6  ret_hfa                   return {f32,f32}      -> [2 x f32] (HFA return)
//   F7  arg_hfa_after_7xf64       7 x f64, {f64,f64}    -> [2 x f64] (all on stack)
//
// Group B — 9–16 byte struct args: IntegerPair (always i64+iN on ARM64)
//   B1  arg_int_pair_12           {i32,i32,i32}          → (i64, i32)
//   B2  arg_int_pair_16           {i64,i64}              → (i64, i64)
//   B3  arg_mixed_12_arm64        {f32,f32,i32}          → (i64, i32)  [no SSEPair/Mixed]
//
// Group C — Large struct args (>16 bytes): indirect, NO byval
//   C1  arg_large_indirect_no_byval {i64,i64,i64} → ptr, no byval attribute
//
// Group D — Return value coercion
//   D1  ret_small_int_struct      returns {i32}          → call returns i32
//   D2  ret_int_pair              returns {i32,i32,i32}  → call returns {i64,i32}
//   D3  ret_sret                  returns {i64,i64,i64}  → sret hidden ptr
//
// Group G — Variadic functions: ARM64 Darwin vs Linux split
//   G1  variadic_hfa_linux        Linux:  variadic HFA arg  -> [2 x f32] (VR save area)
//   G2  variadic_float_linux      Linux:  variadic f32 arg  → no bitcast (SIMD reg)
//   G3  variadic_hfa_darwin       Darwin: variadic HFA arg  → integer coercion (GP)
//   G4  variadic_float_darwin     Darwin: variadic f32 arg  → bitcast to i32 (GP)
//   G5  variadic_pair_early_darwin  Darwin: i32, ... {i32,i32,i32} -> [2 x i64]
//   G6  variadic_pair_late_darwin   Darwin: 7 x i64, ... {i32,i32,i32} -> [2 x i64]
//   G7  variadic_align16_darwin     Darwin: i32, ... {i128}  -> i128
//   G8  variadic_pair_linux         Linux:  i32, ... {i32,i32,i32} -> (i64, i32)
//
// Group I - Narrow integer extension (MOCO-5051): Darwin vs Linux split
//   I1  arg_narrow_ints_darwin    Darwin: ui8, si16, bool args -> zeroext, signext, zeroext
//   I2  arg_narrow_ints_linux     Linux:  ui8, si16 args       -> no extension
//   I3  arg_single_field_structs_darwin
//                                 Darwin: {si8}, {ui16}, {bool} args, {si8} result
//                                         -> i8, i16, i8, i8, all unmarked
//
// MODULE HEADER NOTE: Each test uses the standard ARM64 Linux data layout. The data_layout string
// drives struct size and alignment computation. The triple selects
// AAPCSABIInfo.

//===----------------------------------------------------------------------===//
// Group A: Small integer struct arguments (1–8 bytes)
//
// ARM64 AAPCS and x86-64 System V share the same integer coercion rules for
// small non-HFA structs. These tests confirm ARM64 applies them correctly.
//===----------------------------------------------------------------------===//

// A1: 1-byte struct {i8} → i8.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_1byte_int_struct
kgen.func @arg_1byte_int_struct(%arg0: !kgen.struct<(scalar<si8>)>) {
  // CHECK: llvm.call @c_a1(%{{.*}}) : (i8) -> ()
  pop.external_call @c_a1(%arg0) : (!kgen.struct<(scalar<si8>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a1(i8)
}

// -----

// A2: 2-byte struct {i8, i8} → i16.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_2byte_int_struct
kgen.func @arg_2byte_int_struct(%arg0: !kgen.struct<(scalar<si8>, scalar<si8>)>) {
  // CHECK: llvm.call @c_a2(%{{.*}}) : (i16) -> ()
  pop.external_call @c_a2(%arg0) : (!kgen.struct<(scalar<si8>, scalar<si8>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a2(i16)
}

// -----

// A3: 3-byte struct {i8, i8, i8} → i32 (rounds up).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_3byte_int_struct
kgen.func @arg_3byte_int_struct(%arg0: !kgen.struct<(scalar<si8>, scalar<si8>, scalar<si8>)>) {
  // CHECK: llvm.call @c_a3(%{{.*}}) : (i32) -> ()
  pop.external_call @c_a3(%arg0) : (!kgen.struct<(scalar<si8>, scalar<si8>, scalar<si8>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a3(i32)
}

// -----

// A4: 8-byte struct {i32, i32} → i64.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_8byte_int_struct
kgen.func @arg_8byte_int_struct(%arg0: !kgen.struct<(scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_a4(%{{.*}}) : (i64) -> ()
  pop.external_call @c_a4(%arg0) : (!kgen.struct<(scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a4(i64)
}

//===----------------------------------------------------------------------===//
// Group F: HFA (Homogeneous Float Aggregate)
//
// AAPCS rule: a struct of 1-4 members of the same float type, with no
// padding, is an HFA and uses SIMD registers (V0-V3). The lowering coerces
// an HFA to an array of its base type, as clang does. The backend keeps an
// array argument in consecutive registers: if the free SIMD registers are
// too few, all of the array goes on the stack (AAPCS64 C.3). It does not do
// this for a struct argument.
//
// HFA detection (classifyHomogeneousAggregate):
//   - All members must be the same float type (f16, f32, or f64)
//   - At most 4 members (5+ is not HFA)
//   - Mixed int+float or different float types -> not HFA
//
// On Darwin, a variadic HFA arg is not treated as an HFA (see Group G).
//===----------------------------------------------------------------------===//

// -----

// F1: {f32, f32} (8 bytes, all-float, 2 fields) -> HFA -> [2 x f32].
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_hfa_2xf32
kgen.func @arg_hfa_2xf32(%arg0: !kgen.struct<(scalar<f32>, scalar<f32>)>) {
  // CHECK: llvm.call @c_f1(%{{.*}}) : (!llvm.array<2 x f32>) -> ()
  pop.external_call @c_f1(%arg0) : (!kgen.struct<(scalar<f32>, scalar<f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_f1(!llvm.array<2 x f32>)
}

// -----

// F2: {f32, f32, f32, f32} (16 bytes, all-float, 4 fields) -> HFA ->
// [4 x f32]. 4 is the maximum field count for HFA.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_hfa_4xf32
kgen.func @arg_hfa_4xf32(%arg0: !kgen.struct<(scalar<f32>, scalar<f32>, scalar<f32>, scalar<f32>)>) {
  // CHECK: llvm.call @c_f2(%{{.*}}) : (!llvm.array<4 x f32>) -> ()
  pop.external_call @c_f2(%arg0) : (!kgen.struct<(scalar<f32>, scalar<f32>, scalar<f32>, scalar<f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_f2(!llvm.array<4 x f32>)
}

// -----

// F3: {f64, f64} (16 bytes, all-float doubles, 2 fields) -> HFA -> [2 x f64].
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_hfa_2xf64
kgen.func @arg_hfa_2xf64(%arg0: !kgen.struct<(scalar<f64>, scalar<f64>)>) {
  // CHECK: llvm.call @c_f3(%{{.*}}) : (!llvm.array<2 x f64>) -> ()
  pop.external_call @c_f3(%arg0) : (!kgen.struct<(scalar<f64>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_f3(!llvm.array<2 x f64>)
}

// -----

// F4: {f32 × 5} (20 bytes) → NOT HFA (5 fields exceeds the 4-field limit) AND
// size >16 bytes → indirect. classifyHomogeneousAggregate rejects 5-field
// structs, so the struct falls through to size classification (>16 → indirect).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_not_hfa_5xf32
kgen.func @arg_not_hfa_5xf32(%arg0: !kgen.struct<(scalar<f32>, scalar<f32>, scalar<f32>, scalar<f32>, scalar<f32>)>) {
  // CHECK: llvm.call @c_f4(%{{.*}}) : (!llvm.ptr) -> ()
  pop.external_call @c_f4(%arg0) : (!kgen.struct<(scalar<f32>, scalar<f32>, scalar<f32>, scalar<f32>, scalar<f32>)>) -> ()
  hlcf.return
}
}

// -----

// F5: {f32, i32} (8 bytes) → NOT HFA (mixed int+float) → integer coercion → i64.
// classifyHomogeneousAggregate rejects mixed types, so the struct is classified
// as a non-HFA integer struct.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_not_hfa_mixed
kgen.func @arg_not_hfa_mixed(%arg0: !kgen.struct<(scalar<f32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_f5(%{{.*}}) : (i64) -> ()
  // CHECK-NOT: llvm.call @c_f5(%{{.*}}) : (!llvm.struct
  pop.external_call @c_f5(%arg0) : (!kgen.struct<(scalar<f32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_f5(i64)
}

// -----

// F6: return {f32, f32} -> HFA -> [2 x f32]. clang returns an HFA as its
// struct type, but V0-V3 always hold an HFA result, so the registers are the
// same.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_hfa
kgen.func @ret_hfa() {
  // CHECK: llvm.call @c_f6() : () -> !llvm.array<2 x f32>
  %r = pop.external_call @c_f6() : () -> !kgen.struct<(scalar<f32>, scalar<f32>)>
  hlcf.return
}
// CHECK: llvm.func @c_f6() -> !llvm.array<2 x f32>
}

// -----

// F7: {f64, f64} after seven f64 arguments -> HFA -> [2 x f64]. Only V7 is
// free, so all of the HFA goes on the stack (AAPCS64 C.3). The backend does
// this for an array argument; it would put the first member of a struct
// argument in V7.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_hfa_after_7xf64
kgen.func @arg_hfa_after_7xf64(%d: !kgen.scalar<f64>, %s: !kgen.struct<(scalar<f64>, scalar<f64>)>) {
  // CHECK: llvm.call @c_f7(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, f64, f64, f64, f64, f64, !llvm.array<2 x f64>) -> ()
  pop.external_call @c_f7(%d, %d, %d, %d, %d, %d, %d, %s) : (!kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.struct<(scalar<f64>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_f7(f64, f64, f64, f64, f64, f64, f64, !llvm.array<2 x f64>)
}

//===----------------------------------------------------------------------===//
// Group B: Two-register struct arguments (9–16 bytes)
//
// ARM64 AAPCS always uses IntegerPair for non-HFA structs in this range.
// There is no SSEPair or Mixed class — both registers are always integer
// (i64 for the first 8 bytes, iN for the remaining bytes). This contrasts
// with x86-64 which classifies each eightbyte independently.
//===----------------------------------------------------------------------===//

// -----

// B1: {i32, i32, i32} (12 bytes) → IntegerPair → (i64, i32).
// ARM64: first 8 bytes → i64, remaining 4 bytes → i32.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_int_pair_12
kgen.func @arg_int_pair_12(%arg0: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_b1(%{{.*}}, %{{.*}}) : (i64, i32) -> ()
  pop.external_call @c_b1(%arg0) : (!kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_b1(i64, i32)
}

// -----

// B2: {i64, i64} (16 bytes) → IntegerPair → (i64, i64).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_int_pair_16
kgen.func @arg_int_pair_16(%arg0: !kgen.struct<(scalar<si64>, scalar<si64>)>) {
  // CHECK: llvm.call @c_b2(%{{.*}}, %{{.*}}) : (i64, i64) -> ()
  pop.external_call @c_b2(%arg0) : (!kgen.struct<(scalar<si64>, scalar<si64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_b2(i64, i64)
}

// -----

// B3: {f32, f32, i32} (12 bytes) → IntegerPair → (i64, i32).
// This struct has float fields but it is NOT HFA (mixed types). ARM64 always
// uses IntegerPair for non-HFA 9–16 byte structs, regardless of field types.
// Compare with x86-64 which classifies each eightbyte and could produce Mixed.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_mixed_12_arm64
kgen.func @arg_mixed_12_arm64(%arg0: !kgen.struct<(scalar<f32>, scalar<f32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_b3(%{{.*}}, %{{.*}}) : (i64, i32) -> ()
  // CHECK-NOT: llvm.call @c_b3(%{{.*}}) : (f64
  pop.external_call @c_b3(%arg0) : (!kgen.struct<(scalar<f32>, scalar<f32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_b3(i64, i32)
}

//===----------------------------------------------------------------------===//
// Group C: Large struct arguments (>16 bytes) — indirect, NO byval
//
// ARM64 AAPCS: structs >16 bytes are passed by pointer (X8 register).
// Unlike x86-64, ARM64 does NOT use the byval attribute. The pointer is
// passed directly without any stack copy annotation. addByvalAttrsToFunc
// is only called when triple.isX86().
//===----------------------------------------------------------------------===//

// -----

// C1: {i64, i64, i64} (24 bytes) → indirect via pointer, no byval.
// The function declaration takes !llvm.ptr WITHOUT byval. This is the key
// behavioral difference from x86-64 (see extern-c-abi-x86-64.mlir test C1).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_large_indirect_no_byval
kgen.func @arg_large_indirect_no_byval(%arg0: !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>) {
  // CHECK: llvm.call @c_c1(%{{.*}}) : (!llvm.ptr) -> ()
  pop.external_call @c_c1(%arg0) : (!kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>) -> ()
  hlcf.return
}
// The function takes a plain ptr — no byval attribute.
// CHECK: llvm.func @c_c1(!llvm.ptr)
// CHECK-NOT: llvm.byval
}

//===----------------------------------------------------------------------===//
// Group D: Return value coercion
//===----------------------------------------------------------------------===//

// -----

// D1: return {i32} (4 bytes) → call returns i32, bitcast back to struct.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_small_int_struct
kgen.func @ret_small_int_struct() {
  // CHECK: [[R:%.*]] = llvm.call @c_d1() : () -> i32
  // CHECK: llvm.store [[R]], %{{.*}} : i32, !llvm.ptr
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i32)>
  %r = pop.external_call @c_d1() : () -> !kgen.struct<(scalar<si32>)>
  hlcf.return
}
// CHECK: llvm.func @c_d1() -> i32
}

// -----

// D2: return {i32, i32, i32} (12 bytes) → IntegerPair → call returns
// !llvm.struct<(i64, i32)>. Two-register reconstruction.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_int_pair
kgen.func @ret_int_pair() {
  // CHECK: [[R:%.*]] = llvm.call @c_d2() : () -> !llvm.struct<(i64, i32)>
  // CHECK: llvm.store [[R]], [[SLOT:%.*]] : !llvm.struct<(i64, i32)>, !llvm.ptr
  // CHECK: llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i32)>
  %r = pop.external_call @c_d2() : () -> !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>
  hlcf.return
}
// CHECK: llvm.func @c_d2() -> !llvm.struct<(i64, i32)>
}

// -----

// D3: return {i64, i64, i64} (24 bytes) → sret.
// ARM64 uses X8 for the sret pointer (same calling convention shape as x86-64
// for large returns, though the register differs at the hardware level).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_sret
kgen.func @ret_sret() {
  // CHECK: llvm.call @c_d3(%{{.*}}) : (!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}) -> ()
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i64, i64, i64)>
  %r = pop.external_call @c_d3() : () -> !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>
  hlcf.return
}
// CHECK: llvm.func @c_d3(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>})
}

//===----------------------------------------------------------------------===//
// Group G: Variadic functions — Darwin vs Linux ARM64 split
//
// ARM64 has two different va_list layouts depending on the OS:
//
// Darwin (macOS/iOS): va_list is a flat pointer into a contiguous GP/stack
//   area. All variadic args — including HFA structs and floats — must land in
//   GPRs. The lowering coerces HFA structs to integer and bitcasts f32→i32 to
//   prevent LLVM float→double promotion from putting them in SIMD registers.
//
// Linux AAPCS64: va_list has separate GP (__gr_top) and VR (__vr_top) save
//   areas. HFA variadic args and float args land in SIMD/VR registers, and
//   va_arg for HFA structs reads from the VR save area. The lowering
//   classifies HFAs as for fixed args and does not bitcast floats, so LLVM
//   places them in SIMD registers.
//
// References:
//   AAPCS64 spec IHI0055 §B.4 — va_list structure definition
//   clang/lib/CodeGen/Targets/AArch64.cpp — EmitDarwinVAArg vs EmitAAPCSVAArg
//===----------------------------------------------------------------------===//

// -----

// G1: Linux ARM64 - variadic HFA arg is an HFA.
// {f32, f32} is HFA. On Linux, a variadic HFA arg is classified as a fixed
// one is, so LLVM places it in SIMD registers where Linux va_arg for HFA
// structs reads from the VR save area.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_hfa_linux
kgen.func @variadic_hfa_linux(%fixed: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<f32>, scalar<f32>)>) {
  // The HFA is coerced to an array of f32, not to i64.
  // CHECK: llvm.call @c_g1(%{{.*}}, %{{.*}}) : (i32, !llvm.array<2 x f32>) -> ()
  // CHECK-NOT: llvm.call @c_g1(%{{.*}}, %{{.*}}) : (i32, i64)
  pop.external_call @c_g1(%fixed, %s)
    : (!kgen.scalar<si32>, !kgen.struct<(scalar<f32>, scalar<f32>)>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g1(i32, ...)
}

// -----

// G2: Linux ARM64 — variadic f32 arg, no bitcast.
// On Linux, LLVM places f32 in SIMD registers for variadic calls. Linux va_arg
// for float structs reads from the VR save area, so the bits arrive correctly
// without any bitcast. No applyARM64VariadicFloatBitcast on Linux.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_float_linux
kgen.func @variadic_float_linux(%fixed: !kgen.scalar<si32>, %f: !kgen.scalar<f32>) {
  // No bitcast on Linux — f32 is passed as f32 directly.
  // CHECK-NOT: llvm.bitcast %{{.*}} : f32 to i32
  // CHECK: llvm.call @c_g2(%{{.*}}, %{{.*}}) : (i32, f32) -> ()
  pop.external_call @c_g2(%fixed, %f)
    : (!kgen.scalar<si32>, !kgen.scalar<f32>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g2(i32, ...)
}

// -----

// G3: Darwin ARM64 — variadic HFA arg coerced to integer.
// {f32, f32} is HFA. On Darwin, variadic HFA args must use GPRs because
// Darwin's flat va_list has no separate VR save area. The isVariadicArg +
// isDarwin flag suppresses the HFA classification, and the struct falls
// through to size classification: 8 bytes → i64.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_hfa_darwin
kgen.func @variadic_hfa_darwin(%fixed: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<f32>, scalar<f32>)>) {
  // HFA struct coerced to i64 on Darwin so va_arg reads from GP save area.
  // CHECK: llvm.call @c_g3(%{{.*}}, %{{.*}}) : (i32, i64) -> ()
  // CHECK-NOT: llvm.call @c_g3(%{{.*}}, %{{.*}}) : (i32, !llvm.struct
  pop.external_call @c_g3(%fixed, %s)
    : (!kgen.scalar<si32>, !kgen.struct<(scalar<f32>, scalar<f32>)>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g3(i32, ...)
}

// -----

// G4: Darwin ARM64 — variadic f32 arg bitcast to i32.
// On Darwin, LLVM would otherwise promote f32 to f64 for variadic calls
// (standard C float→double promotion), placing the value in a SIMD register.
// Darwin's flat va_list reads from GPRs only, so bits would be lost. The
// applyARM64VariadicFloatBitcast helper bitcasts f32→i32 to force bits into GPR.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_float_darwin
kgen.func @variadic_float_darwin(%fixed: !kgen.scalar<si32>, %f: !kgen.scalar<f32>) {
  // CHECK: [[BITS:%.*]] = llvm.bitcast %{{.*}} : f32 to i32
  // CHECK: llvm.call @c_g4(%{{.*}}, [[BITS]])
  pop.external_call @c_g4(%fixed, %f)
    : (!kgen.scalar<si32>, !kgen.scalar<f32>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g4(i32, ...)
}

// -----

// G5: Darwin ARM64 - variadic 9-16 byte composite with free registers.
// Apple's arm64 ABI passes variadic arguments on the stack, so the composite
// takes clang's single-integer form even though X1-X7 are free.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_pair_early_darwin
kgen.func @variadic_pair_early_darwin(%fixed: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_g5(%{{.*}}, %{{.*}}) : (i32, !llvm.array<2 x i64>) -> ()
  pop.external_call @c_g5(%fixed, %s)
    : (!kgen.scalar<si32>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g5(i32, ...)
}

// -----

// G6: Darwin ARM64 - the same composite after seven fixed args has the same
// form as in G5.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_pair_late_darwin
kgen.func @variadic_pair_late_darwin(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %g: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_g6(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64, !llvm.array<2 x i64>) -> ()
  pop.external_call @c_g6(%a, %b, %c, %d, %e, %f, %g, %s)
    : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> () <numFixedArgs = 7 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g6(i64, i64, i64, i64, i64, i64, i64, ...)
}

// -----

// G7: Darwin ARM64 - a 16-byte-aligned variadic composite becomes i128, so it
// keeps its 16-byte stack-slot alignment.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_align16_darwin
kgen.func @variadic_align16_darwin(%fixed: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<si128>)>) {
  // CHECK: llvm.call @c_g7(%{{.*}}, %{{.*}}) : (i32, i128) -> ()
  pop.external_call @c_g7(%fixed, %s)
    : (!kgen.scalar<si32>, !kgen.struct<(scalar<si128>)>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g7(i32, ...)
}

// -----

// G8: Linux ARM64 - control for G5. Linux AAPCS64 passes variadic arguments
// in X0-X7 like fixed arguments, so the composite stays a two-register pair.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_pair_linux
kgen.func @variadic_pair_linux(%fixed: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_g8(%{{.*}}, %{{.*}}, %{{.*}}) : (i32, i64, i32) -> ()
  pop.external_call @c_g8(%fixed, %s)
    : (!kgen.scalar<si32>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g8(i32, ...)
}

//===----------------------------------------------------------------------===//
// Group I: Narrow integer extension (Darwin vs Linux)
//
// Apple's arm64 ABI requires the caller to extend a narrow integer; AAPCS64
// does not, and clang emits no extension on Linux. The sign
// comes from the POP dtype. Only the arguments are marked: a mark on the result
// would let LLVM assume the callee extended it, but a C struct flattened to a
// bare Mojo integer comes back with unspecified upper bits.
//===----------------------------------------------------------------------===//

// -----

// I1: Darwin - ui8, si16, bool and si32 arguments, si8 result.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_narrow_ints_darwin
kgen.func @arg_narrow_ints_darwin(%u8: !kgen.scalar<ui8>, %s16: !kgen.scalar<si16>, %bool: !kgen.scalar<bool>, %s32: !kgen.scalar<si32>) {
  // CHECK: llvm.call @c_i1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i8 {llvm.zeroext}, i16 {llvm.signext}, i1 {llvm.zeroext}, i32) -> i8
  %r = pop.external_call @c_i1(%u8, %s16, %bool, %s32) : (!kgen.scalar<ui8>, !kgen.scalar<si16>, !kgen.scalar<bool>, !kgen.scalar<si32>) -> !kgen.scalar<si8>
  hlcf.return
}
// CHECK: llvm.func @c_i1(i8 {llvm.zeroext}, i16 {llvm.signext}, i1 {llvm.zeroext}, i32) -> i8
}

// -----

// I2: Linux - the same signature gets no extension.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_narrow_ints_linux
kgen.func @arg_narrow_ints_linux(%u8: !kgen.scalar<ui8>, %s16: !kgen.scalar<si16>) {
  // CHECK: llvm.call @c_i2(%{{.*}}, %{{.*}}) : (i8, i16) -> i8
  %r = pop.external_call @c_i2(%u8, %s16) : (!kgen.scalar<ui8>, !kgen.scalar<si16>) -> !kgen.scalar<si8>
  hlcf.return
}
// CHECK: llvm.func @c_i2(i8, i16) -> i8
}

// -----

// I3: Darwin - {si8}, {ui16} and {bool} arguments, {si8} result. clang
// coerces a struct of one narrow integer to an integer of its size and does
// not extend it, so the coerced integers get no mark, whatever the sign of the
// field.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_single_field_structs_darwin
kgen.func @arg_single_field_structs_darwin(%s8: !kgen.struct<(scalar<si8>)>, %u16: !kgen.struct<(scalar<ui16>)>, %bool: !kgen.struct<(scalar<bool>)>) {
  // CHECK: llvm.call @c_i3(%{{.*}}, %{{.*}}, %{{.*}}) : (i8, i16, i8) -> i8
  %r = pop.external_call @c_i3(%s8, %u16, %bool) : (!kgen.struct<(scalar<si8>)>, !kgen.struct<(scalar<ui16>)>, !kgen.struct<(scalar<bool>)>) -> !kgen.struct<(scalar<si8>)>
  hlcf.return
}
// CHECK: llvm.func @c_i3(i8, i16, i8) -> i8
}

// -----

// Group R — Rollback-to-stack: AAPCS64 passes a composite that does not fit
// in the remaining argument registers entirely in memory, never split across
// a register and the stack. X0-X7 are eight GPRs; an sret return uses X8 and
// does not reduce the budget (contrast the x86-64 file, where sret takes RDI).
//
//   R1  arg_rollback_after_seven    7×i64 + {i32,i32,i32}  → [2 x i64] (X7 alone)
//   R2  arg_no_rollback_after_six   6×i64 + {i32,i32,i32}  → IntegerPair
//   R3  arg_sret_rollback           sret + 7×i64 + struct  → [2 x i64]
//   R4  arg_sret_no_rollback       sret + 6×i64 + struct  → IntegerPair

// R1: seven i64 args consume X0-X6; the 12-byte struct needs two registers
// and only X7 remains, so it rolls back to memory: coerced to [2 x i64] by value.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_rollback_after_seven
kgen.func @arg_rollback_after_seven(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %g: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_ar1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64, !llvm.array<2 x i64>) -> ()
  pop.external_call @c_ar1(%a, %b, %c, %d, %e, %f, %g, %s) : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar1(i64, i64, i64, i64, i64, i64, i64, !llvm.array<2 x i64>)
}

// -----

// R2: control - six i64 args leave two registers, so the same struct fits and
// stays a two-register IntegerPair (i64, i32).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_no_rollback_after_six
kgen.func @arg_no_rollback_after_six(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_ar2(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i32) -> ()
  pop.external_call @c_ar2(%a, %b, %c, %d, %e, %f, %s) : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar2(i64, i64, i64, i64, i64, i64, i64, i32)
}

// -----

// R3: an sret return does not free X0-X7, so seven i64 args still leave only
// X7 and the struct rolls back to memory.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sret_rollback
kgen.func @arg_sret_rollback(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %g: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  %r = pop.external_call @c_ar3(%a, %b, %c, %d, %e, %f, %g, %s) : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>
  hlcf.return
}
// CHECK: llvm.func @c_ar3(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, i64, i64, i64, i64, i64, i64, i64, !llvm.array<2 x i64>)
}

// -----

// R4: control - sret plus six i64 args leave two registers, so the struct
// still fits as IntegerPair. On x86-64 the same shape rolls back because
// sret takes one of the six integer registers.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sret_no_rollback
kgen.func @arg_sret_no_rollback(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  %r = pop.external_call @c_ar4(%a, %b, %c, %d, %e, %f, %s) : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>
  hlcf.return
}
// CHECK: llvm.func @c_ar4(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, i64, i64, i64, i64, i64, i64, i64, i32)
}

// -----

// R5: the rolled-back struct exhausts the register search, so a later
// argument goes to the stack too (clang does the same). The trailing i64
// stays a direct parameter after the [2 x i64] aggregate.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_after_rollback
kgen.func @arg_after_rollback(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %g: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>, %t: !kgen.scalar<si64>) {
  // CHECK: llvm.call @c_ar5(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64, !llvm.array<2 x i64>, i64) -> ()
  pop.external_call @c_ar5(%a, %b, %c, %d, %e, %f, %g, %s, %t) : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>, !kgen.scalar<si64>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar5(i64, i64, i64, i64, i64, i64, i64, !llvm.array<2 x i64>, i64)
}

// -----

// R6: a 16-byte-aligned composite rolls back to i128, not [2 x i64], keeping
// its 16-byte stack-slot alignment (clang's rule for aligned aggregates).
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_align16_rollback
kgen.func @arg_align16_rollback(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %g: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si128>)>) {
  // CHECK: llvm.call @c_ar6(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i128) -> ()
  pop.external_call @c_ar6(%a, %b, %c, %d, %e, %f, %g, %s) : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si128>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar6(i64, i64, i64, i64, i64, i64, i64, i128)
}

// -----

// R7: a 16-byte-aligned composite that fits in registers is an i128 too, so
// the backend starts it at an even register (X2), as clang does.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_align16_in_registers
kgen.func @arg_align16_in_registers(%a: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si128>)>) {
  // CHECK: llvm.call @c_ar7(%{{.*}}, %{{.*}}) : (i64, i128) -> ()
  pop.external_call @c_ar7(%a, %s) : (!kgen.scalar<si64>, !kgen.struct<(scalar<si128>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar7(i64, i128)
}

// -----

// R8: the i128 of R7 skips X1, so three more i64 leave only X7 and the
// 12-byte struct rolls back to memory.
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_align16_skip_then_rollback
kgen.func @arg_align16_skip_then_rollback(%a: !kgen.scalar<si64>, %w: !kgen.struct<(scalar<si128>)>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_ar8(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i128, i64, i64, i64, !llvm.array<2 x i64>) -> ()
  pop.external_call @c_ar8(%a, %w, %b, %c, %d, %s) : (!kgen.scalar<si64>, !kgen.struct<(scalar<si128>)>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar8(i64, i128, i64, i64, i64, !llvm.array<2 x i64>)
}

// -----

// R9: Darwin control for R8. Apple's arm64 ABI does not skip X1, so the
// 12-byte struct still fits in X6 and X7.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_align16_no_skip_darwin
kgen.func @arg_align16_no_skip_darwin(%a: !kgen.scalar<si64>, %w: !kgen.struct<(scalar<si128>)>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_ar9(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i128, i64, i64, i64, i64, i32) -> ()
  pop.external_call @c_ar9(%a, %w, %b, %c, %d, %s) : (!kgen.scalar<si64>, !kgen.struct<(scalar<si128>)>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar9(i64, i128, i64, i64, i64, i64, i32)
}

// -----

// R10: an `@align(16)` struct of two si64 has 16-byte alignment, but its
// fields have only 8-byte alignment. Apple's arm64 ABI uses the alignment of
// the type, so the struct rolls back to i128 and keeps its 16-byte stack slot.
module attributes {M.target_info = #M.target<triple="aarch64-apple-macosx12.0.0", arch="", features="", data_layout="e-m:o-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_decorated_align16_rollback_darwin
kgen.func @arg_decorated_align16_rollback_darwin(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %g: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si64>, scalar<si64>) align(16)>) {
  // CHECK: llvm.call @c_ar10(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i128) -> ()
  pop.external_call @c_ar10(%a, %b, %c, %d, %e, %f, %g, %s) : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si64>, scalar<si64>) align(16)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar10(i64, i64, i64, i64, i64, i64, i64, i128)
}

// -----

// R11: Linux control for R10. AAPCS64 uses the natural alignment, which does
// not include `@align`, so the struct rolls back to [2 x i64].
module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_decorated_align16_rollback_linux
kgen.func @arg_decorated_align16_rollback_linux(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %f: !kgen.scalar<si64>, %g: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si64>, scalar<si64>) align(16)>) {
  // CHECK: llvm.call @c_ar11(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64, !llvm.array<2 x i64>) -> ()
  pop.external_call @c_ar11(%a, %b, %c, %d, %e, %f, %g, %s) : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si64>, scalar<si64>) align(16)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_ar11(i64, i64, i64, i64, i64, i64, i64, !llvm.array<2 x i64>)
}
