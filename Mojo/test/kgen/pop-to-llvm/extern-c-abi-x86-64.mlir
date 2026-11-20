// RUN: kgen-opt -lower-kgen-to-llvm -lower-global-pop-to-llvm -split-input-file %s | FileCheck %s

// Unit tests for C ABI lowering on x86-64 (System V AMD64 ABI).
//
// PURPOSE: These are UNIT tests that lock the current behavior of the lowering
// transform. They exist to catch regressions during refactoring. They may need
// updating if we discover ABI bugs or add new features. The system tests are
// the authoritative check for correct end-to-end C ABI behavior:
//   KGEN/test/mojo-integration/extern-c-abi/
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/LowerPOPToLLVMExternalCalls.cpp — ConvertPOPExternalCall
//   Mojo/lib/KGENToLLVM/CABISystemV.cpp                 — SystemVABIInfo classifier
//   Mojo/lib/KGENToLLVM/CABILowering.cpp                — shared utilities
//
// RELATED TEST FILES:
//   extern-c-abi.mlir         — platform-independent tests (DefaultCABIInfo,
//                               empty triple): struct not field-exploded,
//                               multi-value return, identity pass-through
//   extern-c-abi-aarch64.mlir — ARM64 AAPCS tests
//
// INPUT FORMAT NOTE: Each test uses kgen.func as the enclosing function. The
// aggregate operands are KGEN types (!kgen.struct, !pop.array), as the Mojo
// compiler emits them. The classifier reads the KGEN type, which gives the
// field offsets, the alignment and the signedness. -lower-kgen-to-llvm
// changes kgen.func to llvm.func before -lower-global-pop-to-llvm lowers
// pop.external_call, because createEntryBlockAlloca requires an
// LLVM::LLVMFuncOp as the enclosing function.
//
// TEST TABLE:
// Group A — Small struct args (1–8 bytes): Integer coercion
//   A1  arg_1byte_int_struct      {i8}        → i8
//   A2  arg_2byte_int_struct      {i8,i8}     → i16
//   A3  arg_3byte_int_struct      {i8,i8,i8}  → i32  (rounds up)
//   A4  arg_8byte_int_struct      {i32,i32}   → i64
//   A5  arg_padded_struct         {i32,i8}    → i64  (padded to 8 bytes)
//
// Group E — Small all-float struct args (1–8 bytes): SSE coercion
//   E1  arg_float_struct          {f32}       → f32
//   E2  arg_double_struct         {f64}       → f64
//   E3  arg_two_floats            {f32,f32}   → f64  (fits one SSE register)
//   E4  arg_int_wins_float        {i32,f32}   → i64  (INTEGER overrides SSE)
//
// Group B — 9–16 byte struct args: two-register classification
//   B1  arg_int_pair_12          {i32,i32,i32}        → (i64, i32)  IntegerPair
//   B2  arg_int_pair_16          {i64,i64}            → (i64, i64)  IntegerPair
//   B3  arg_sse_pair_12          {f32,f32,f32}        → (f64, f32)  SSEPair
//   B4  arg_sse_pair_16          {f64,f64}            → (f64, f64)  SSEPair
//   B5  arg_mixed_16             {i32,i32,f64}        → (i64, f64)  Mixed
//   B6  arg_dense_int_16         {i8,i8,i16,i32,i64}  → (i64, i64)  IntegerPair
//         (many integer fields per eightbyte, all classify as INTEGER)
//
// Group C — Large struct args (>16 bytes): indirect pass + byval
//   C1  arg_large_indirect        {i64,i64,i64}       → ptr with byval on func
//   C2  arg_padded_indirect       {i8,f32,i8,f64}     → ptr with byval on func
//         (alignment padding expands to 24 bytes → MEMORY class)
//
// Group D — Return value coercion
//   D1  ret_small_int_struct      returns {i32}       → call returns i32
//   D2  ret_int_pair              returns {i32,i32,i32} → call returns {i64,i32}
//   D3  ret_sse_pair              returns {f64,f64}   → call returns {f64,f64}
//   D4  ret_sret                  returns {i64,i64,i64} → sret hidden ptr
//
// Group G — Variadic functions
//   G1  variadic_scalars          variadic call, scalar args  → func has ...
//   G2  variadic_large_struct     variadic large struct arg   → byval on call
//
// Group H — Attribute remapping
//   H1  func_attrs_passthrough    funcAttrs = ["noinline"]    → passthrough on llvm.func
//   H2  arg_attrs_two_reg         argAttrs on two-reg struct  -> no noundef on the struct
//   H3  arg_attrs_sret            argAttrs with sret return   → attr shifted past sret ptr
//   H4  res_attrs_struct          resAttrs on a struct result -> no noundef on the result
//
// Group I - Narrow integer extension (MOCO-5051)
//   I1  arg_narrow_ints           ui8, si16, bool, si32 args  -> zeroext, signext, zeroext, none
//                                 on the call and the declaration; si8 result unmarked
//   I2  arg_single_field_structs  {si8}, {ui16}, {bool} args  -> i8, i16, i8, unmarked
//                                 {si8} result                -> i8, unmarked
//
// Group F — Vector type fields: classified by vector size
//   F1  arg_vec2xf32_struct       {vector<2xf32>}       correct: f64 (SSE)
//   F2  arg_vec2xf32_and_f64      {vector<2xf32>,f64}   correct: SSEPair(f64,f64)
//   F3  arg_vec4xf32_struct       {vector<4xf32>}       correct: vector<2xf64> (SSE+SSEUP)
//   F4  arg_vec2xi32_struct       {vector<2xi32>}       correct: f64 (SSE)
//
// Group N — Nested struct and array fields
//   N1  arg_nested_float_4        {struct<(f32)>}          correct: f32  (SSE)
//   N2  arg_nested_float_8        {struct<(f32,f32)>}      correct: f64  (SSE)
//   N3  arg_nested_float_and_f64  {struct<(f32,f32)>,f64}  correct: SSEPair(f64,f64)
//   N4  arg_nested_int_8          {struct<(i32,i32)>}      correct: i64  (regression guard)
//   N5  arg_nested_int_and_i64    {struct<(i32,i32)>,i64}  correct: IntegerPair(i64,i64)
//   N6  arg_array_member          {[3 x f32]}              correct: SSEPair(f64,f32)
//   N7  arg_empty_member          {f32,{},f32}             correct: f64  (SSE)
//   N8  arg_zero_array_member     {f32,{[0 x f32]},f32}    correct: f64  (SSE)
//
// MODULE HEADER NOTE: Each test uses the standard x86-64 Linux data layout
// string to drive struct size and alignment computation in getStructSize(),
// which in turn determines which ABI classification rule applies. The triple
// selects SystemVABIInfo.

// Reused module header (copy-pasted per test because -split-input-file
// requires a full module per section):
//   triple="x86_64-unknown-linux-gnu"
//   data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

//===----------------------------------------------------------------------===//
// Group A: Small integer struct arguments (1–8 bytes)
//
// Rule: non-HFA structs ≤8 bytes are coerced to the smallest integer type that
// holds their byte count. The struct is bitcast via a store/load pair through
// an entry-block alloca. The LLVM call and function declaration use the
// coerced integer type, not the struct type.
//===----------------------------------------------------------------------===//

// A1: 1-byte struct {i8} → passes in a single byte register (i8).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_1byte_int_struct
kgen.func @arg_1byte_int_struct(%arg0: !kgen.struct<(scalar<si8>)>) {
  // CHECK: llvm.call @c_a1(%{{.*}}) : (i8) -> ()
  pop.external_call @c_a1(%arg0) : (!kgen.struct<(scalar<si8>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a1(i8)
}

// -----

// A2: 2-byte struct {i8, i8} → coerces to i16.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_2byte_int_struct
kgen.func @arg_2byte_int_struct(%arg0: !kgen.struct<(scalar<si8>, scalar<si8>)>) {
  // CHECK: llvm.call @c_a2(%{{.*}}) : (i16) -> ()
  pop.external_call @c_a2(%arg0) : (!kgen.struct<(scalar<si8>, scalar<si8>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a2(i16)
}

// -----

// A3: 3-byte struct {i8, i8, i8} → coerces to i32 (3 bytes rounds up to i32).
// The alloca is sized to i32 (4 bytes) to ensure the load reads valid memory.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_3byte_int_struct
kgen.func @arg_3byte_int_struct(%arg0: !kgen.struct<(scalar<si8>, scalar<si8>, scalar<si8>)>) {
  // CHECK: llvm.call @c_a3(%{{.*}}) : (i32) -> ()
  pop.external_call @c_a3(%arg0) : (!kgen.struct<(scalar<si8>, scalar<si8>, scalar<si8>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a3(i32)
}

// -----

// A4: 8-byte struct {i32, i32} → coerces to i64 (fits in one register).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_8byte_int_struct
kgen.func @arg_8byte_int_struct(%arg0: !kgen.struct<(scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_a4(%{{.*}}) : (i64) -> ()
  pop.external_call @c_a4(%arg0) : (!kgen.struct<(scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a4(i64)
}

// -----

// A5: struct {i32, i8} — the layout pads it to 8 bytes (i32 align=4, i8 at
// offset 4, 3 bytes padding). getStructSize returns 8, so coercion yields i64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_padded_struct
kgen.func @arg_padded_struct(%arg0: !kgen.struct<(scalar<si32>, scalar<si8>)>) {
  // CHECK: llvm.call @c_a5(%{{.*}}) : (i64) -> ()
  pop.external_call @c_a5(%arg0) : (!kgen.struct<(scalar<si32>, scalar<si8>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_a5(i64)
}

//===----------------------------------------------------------------------===//
// Group E: Small all-float struct arguments (SSE classification)
//
// Rule: on x86-64, an all-float struct ≤8 bytes is classified SSE and coerced
// to f32 (≤4 bytes) or f64 (≤8 bytes). This keeps the value in an XMM
// register. If any field is an integer, the whole eightbyte is classified
// INTEGER (Integer overrides SSE) and the struct coerces to iN instead.
//===----------------------------------------------------------------------===//

// -----

// E1: {f32} (4 bytes, all-float) → SSE → f32.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_float_struct
kgen.func @arg_float_struct(%arg0: !kgen.struct<(scalar<f32>)>) {
  // CHECK: llvm.call @c_e1(%{{.*}}) : (f32) -> ()
  pop.external_call @c_e1(%arg0) : (!kgen.struct<(scalar<f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_e1(f32)
}

// -----

// E2: {f64} (8 bytes, all-float) → SSE → f64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_double_struct
kgen.func @arg_double_struct(%arg0: !kgen.struct<(scalar<f64>)>) {
  // CHECK: llvm.call @c_e2(%{{.*}}) : (f64) -> ()
  pop.external_call @c_e2(%arg0) : (!kgen.struct<(scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_e2(f64)
}

// -----

// E3: {f32, f32} (8 bytes, all-float) → SSE → f64.
// Two floats in one eightbyte: the eightbyte is classified SSE, and because
// the eightbyte is 8 bytes, getEightbyteType returns f64 (not <2 x f32>).
// The store/load bitcast pattern reinterprets the two f32 bits as one f64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_two_floats
kgen.func @arg_two_floats(%arg0: !kgen.struct<(scalar<f32>, scalar<f32>)>) {
  // CHECK: llvm.call @c_e3(%{{.*}}) : (f64) -> ()
  pop.external_call @c_e3(%arg0) : (!kgen.struct<(scalar<f32>, scalar<f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_e3(f64)
}

// -----

// E4: {i32, f32} (8 bytes, mixed int + float) → INTEGER wins over SSE → i64.
// The System V ABI rule: if any field in an eightbyte is INTEGER class, the
// whole eightbyte is INTEGER. An i32 field makes this eightbyte INTEGER even
// though f32 would be SSE. Key test: must NOT coerce to f64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_int_wins_float
kgen.func @arg_int_wins_float(%arg0: !kgen.struct<(scalar<si32>, scalar<f32>)>) {
  // CHECK: llvm.call @c_e4(%{{.*}}) : (i64) -> ()
  // CHECK-NOT: llvm.call @c_e4(%{{.*}}) : (f64) -> ()
  pop.external_call @c_e4(%arg0) : (!kgen.struct<(scalar<si32>, scalar<f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_e4(i64)
}

//===----------------------------------------------------------------------===//
// Group B: Two-register struct arguments (9–16 bytes)
//
// Rule: structs 9–16 bytes are split into two 8-byte eightbytes. Each
// eightbyte is classified independently (INTEGER or SSE), then merged into
// one of: IntegerPair (i64, i64/i32), SSEPair (f64, f64/f32), or Mixed
// (i64, f64). The call passes two arguments; the function signature has two
// parameters.
//===----------------------------------------------------------------------===//

// -----

// B1: {i32, i32, i32} (12 bytes) → IntegerPair → (i64, i32).
// Eightbyte 0 (bytes 0-7): two i32s → INTEGER.
// Eightbyte 1 (bytes 8-11): one i32 → INTEGER.
// Both INTEGER → IntegerPair. Types: i64 for the 8-byte half, i32 for the 4-byte half.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
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
// Each eightbyte holds exactly one i64 → INTEGER.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_int_pair_16
kgen.func @arg_int_pair_16(%arg0: !kgen.struct<(scalar<si64>, scalar<si64>)>) {
  // CHECK: llvm.call @c_b2(%{{.*}}, %{{.*}}) : (i64, i64) -> ()
  pop.external_call @c_b2(%arg0) : (!kgen.struct<(scalar<si64>, scalar<si64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_b2(i64, i64)
}

// -----

// B3: {f32, f32, f32} (12 bytes) → SSEPair → (f64, f32).
// Eightbyte 0 (bytes 0-7): two f32s → SSE; 8-byte region → f64.
// Eightbyte 1 (bytes 8-11): one f32 → SSE; 4-byte region → f32.
// Both SSE → SSEPair.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sse_pair_12
kgen.func @arg_sse_pair_12(%arg0: !kgen.struct<(scalar<f32>, scalar<f32>, scalar<f32>)>) {
  // CHECK: llvm.call @c_b3(%{{.*}}, %{{.*}}) : (f64, f32) -> ()
  pop.external_call @c_b3(%arg0) : (!kgen.struct<(scalar<f32>, scalar<f32>, scalar<f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_b3(f64, f32)
}

// -----

// B4: {f64, f64} (16 bytes) → SSEPair → (f64, f64).
// Each eightbyte holds exactly one f64 → SSE.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sse_pair_16
kgen.func @arg_sse_pair_16(%arg0: !kgen.struct<(scalar<f64>, scalar<f64>)>) {
  // CHECK: llvm.call @c_b4(%{{.*}}, %{{.*}}) : (f64, f64) -> ()
  pop.external_call @c_b4(%arg0) : (!kgen.struct<(scalar<f64>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_b4(f64, f64)
}

// -----

// B5: {i32, i32, f64} (16 bytes) → Mixed → (i64, f64).
// Eightbyte 0 (bytes 0-7): two i32s → INTEGER → i64.
// Eightbyte 1 (bytes 8-15): one f64 → SSE → f64.
// INTEGER + SSE → Mixed.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_mixed_16
kgen.func @arg_mixed_16(%arg0: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<f64>)>) {
  // CHECK: llvm.call @c_b5(%{{.*}}, %{{.*}}) : (i64, f64) -> ()
  pop.external_call @c_b5(%arg0) : (!kgen.struct<(scalar<si32>, scalar<si32>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_b5(i64, f64)
}

//===----------------------------------------------------------------------===//
// Group C: Large struct arguments (>16 bytes) — indirect passing
//
// Rule: structs >16 bytes are passed indirectly. The caller allocates stack
// space, stores the struct there, and passes a pointer. On x86-64 System V,
// the byval attribute is set on the function parameter to tell LLVM the callee
// receives a copy on the stack. ARM64 does NOT use byval (see aarch64 file).
//===----------------------------------------------------------------------===//

// -----

// C1: {i64, i64, i64} (24 bytes) → indirect via pointer, byval on func param.
// The function declaration takes !llvm.ptr with byval. The alloca+store is
// done by prepareCoercedArgument. addByvalAttrsToFunc sets the byval attr.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_large_indirect
kgen.func @arg_large_indirect(%arg0: !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>) {
  // CHECK: llvm.call @c_c1(%{{.*}}) : (!llvm.ptr {llvm.byval = !llvm.struct<(i64, i64, i64)>}) -> ()
  pop.external_call @c_c1(%arg0) : (!kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_c1(!llvm.ptr {llvm.byval = !llvm.struct<(i64, i64, i64)>})
}

//===----------------------------------------------------------------------===//
// Group D: Return value coercion
//
// Same classification rules as arguments, but applied to the return value.
// The LLVM call returns the coerced type; the transform reconstructs the
// original struct with a store and a load through a stack slot. For two
// registers, the stored value is the struct of both. For >16 bytes, sret convention
// is used: a hidden first pointer parameter is allocated by the caller;
// the callee writes the result there; the caller loads it afterward.
//===----------------------------------------------------------------------===//

// -----

// D1: return {i32} (4 bytes) → call returns i32, bitcast back to struct.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
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
// !llvm.struct<(i64, i32)>. Two-register reconstruction: store the pair to a
// stack slot, load as the original struct.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
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

// D3: return {f64, f64} (16 bytes) → SSEPair → call returns
// !llvm.struct<(f64, f64)>. Store the pair and reload it as the struct.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_sse_pair
kgen.func @ret_sse_pair() {
  // CHECK: [[R:%.*]] = llvm.call @c_d3() : () -> !llvm.struct<(f64, f64)>
  // CHECK: llvm.store [[R]], [[SLOT:%.*]] : !llvm.struct<(f64, f64)>, !llvm.ptr
  // CHECK: llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(f64, f64)>
  %r = pop.external_call @c_d3() : () -> !kgen.struct<(scalar<f64>, scalar<f64>)>
  hlcf.return
}
// CHECK: llvm.func @c_d3() -> !llvm.struct<(f64, f64)>
}

// -----

// D4: return {i64, i64, i64} (24 bytes) → sret.
// The function gets a hidden first parameter (!llvm.ptr {llvm.sret = ...}).
// The LLVM call is void. The caller loads the result from the sret pointer.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @ret_sret
kgen.func @ret_sret() {
  // CHECK: llvm.call @c_d4(%{{.*}}) : (!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}) -> ()
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i64, i64, i64)>
  %r = pop.external_call @c_d4() : () -> !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>
  hlcf.return
}
// CHECK: llvm.func @c_d4(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>})
}

//===----------------------------------------------------------------------===//
// Group G: Variadic functions
//
// Variadic functions on x86-64: fixed args follow the same classification
// rules as non-variadic. The function type marks the function as variadic
// (isVarArg=true / '...' in the type). For large struct variadic args, byval
// is set on the *call* instruction (not the function declaration) because
// variadic args are not part of the function signature.
//===----------------------------------------------------------------------===//

// -----

// G1: variadic call with scalar args → function type includes '...'.
// The scalar args pass through as-is (identity classification). The key
// observable is that the LLVM function type is variadic.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_scalars
kgen.func @variadic_scalars(%fixed: !kgen.scalar<si32>, %extra: !kgen.scalar<si32>) {
  // CHECK: llvm.call @c_g1(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
  pop.external_call @c_g1(%fixed, %extra)
    : (!kgen.scalar<si32>, !kgen.scalar<si32>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_g1(i32, ...)
}

// -----

// G2: variadic call with a large struct variadic arg → byval on the call.
// The fixed arg (i32) is declared in the function signature normally. The
// variadic large struct arg (24 bytes, >16 → indirect) is passed as a pointer,
// but byval cannot appear on the function declaration (variadic args are not
// in the signature). Instead, addByvalAttrsToCall sets byval on the call
// instruction itself.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_large_struct
kgen.func @variadic_large_struct(%fixed: !kgen.scalar<si32>,
                                  %s: !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>) {
  // CHECK: llvm.call @c_g2
  // CHECK-SAME: llvm.byval = !llvm.struct<(i64, i64, i64)>
  pop.external_call @c_g2(%fixed, %s)
    : (!kgen.scalar<si32>, !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// The function declaration does NOT have byval — only the call does.
// CHECK: llvm.func @c_g2(i32, ...)
// CHECK-NOT: llvm.byval
}

//===----------------------------------------------------------------------===//
// Group B (continued): Dense integer struct (B6)
//===----------------------------------------------------------------------===//

// -----

// B6: {i8, i8, i16, i32, i64} (16 bytes) → IntegerPair → (i64, i64).
// Eightbyte 0 (bytes 0-7): i8 + i8 + i16 + i32 — four integer fields, all
// INTEGER class. Size=8 → i64.
// Eightbyte 1 (bytes 8-15): i64 — one integer field, INTEGER class. Size=8 → i64.
// Exercises the case where an eightbyte contains multiple fields of different
// integer widths; they should all classify as INTEGER and coerce to i64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_dense_int_16
kgen.func @arg_dense_int_16(%arg0: !kgen.struct<(scalar<si8>, scalar<si8>, scalar<si16>, scalar<si32>, scalar<si64>)>) {
  // CHECK: llvm.call @c_b6(%{{.*}}, %{{.*}}) : (i64, i64) -> ()
  pop.external_call @c_b6(%arg0) : (!kgen.struct<(scalar<si8>, scalar<si8>, scalar<si16>, scalar<si32>, scalar<si64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_b6(i64, i64)
}

//===----------------------------------------------------------------------===//
// Group C (continued): Alignment-padded large struct (C2)
//===----------------------------------------------------------------------===//

// -----

// C2: {i8, f32, i8, f64} → indirect via pointer, byval on func param.
// Layout (x86-64): i8@0, pad@1-3, f32@4, i8@8, pad@9-15, f64@16.
// Total size = 24 bytes (> 16) → MEMORY class → indirect pass with byval.
// This tests that structs whose logical field sizes sum to ≤16 bytes can still
// exceed 16 bytes due to alignment padding, triggering the MEMORY path.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_padded_indirect
kgen.func @arg_padded_indirect(%arg0: !kgen.struct<(scalar<si8>, scalar<f32>, scalar<si8>, scalar<f64>)>) {
  // CHECK: llvm.call @c_c2(%{{.*}}) : (!llvm.ptr {llvm.byval = !llvm.struct<(i8, f32, i8, f64)>}) -> ()
  pop.external_call @c_c2(%arg0) : (!kgen.struct<(scalar<si8>, scalar<f32>, scalar<si8>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_c2(!llvm.ptr {llvm.byval = !llvm.struct<(i8, f32, i8, f64)>})
}

//===----------------------------------------------------------------------===//
// Group H: Attribute remapping through ABI coercion
//
// These tests verify that funcAttrs, argAttrs, and resAttrs on
// pop.external_call are correctly forwarded to the lowered llvm.func, and
// that argAttrs are remapped when ABI coercion changes the number or order of
// parameters (sret insertion, two-register expansion).
//===----------------------------------------------------------------------===//

// -----

// H1: funcAttrs passthrough → LLVM function has passthrough attribute.
// Verifies that funcAttrs = ["noinline"] on the call site are propagated to
// the LLVM function declaration as passthrough = ["noinline"].
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @func_attrs_passthrough
kgen.func @func_attrs_passthrough(%arg0: !kgen.scalar<si32>) {
  // CHECK: llvm.call @c_h1
  pop.external_call @c_h1(%arg0) : (!kgen.scalar<si32>) -> () <funcAttrs = ["noinline"]>
  hlcf.return
}
// CHECK: llvm.func @c_h1
// CHECK-SAME: passthrough = {{.*}}"noinline"
}

// -----

// H2: argAttrs remapped through two-register coercion.
// The POP call has a struct arg and an i32 arg. Both have {llvm.noundef}.
// IntegerPair coercion changes the 12-byte struct to (i64, i32). The struct
// does not keep noundef, because its registers can contain bytes that are
// not defined. The attributes of the i32 arg move to param 2.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_attrs_two_reg
kgen.func @arg_attrs_two_reg(%arg0: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>, %arg1: !kgen.scalar<si32>) {
  // CHECK: llvm.call @c_h2
  pop.external_call @c_h2(%arg0, %arg1)
    : (!kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>, !kgen.scalar<si32>) -> () <argAttrs = [{llvm.noundef}, {llvm.noundef}]>
  hlcf.return
}
// POP arg 0 becomes LLVM params 0 (i64) and 1 (i32). POP arg 1 becomes
// LLVM param 2.
// CHECK: llvm.func @c_h2(i64, i32, i32 {llvm.noundef})
}

// -----

// H3: argAttrs remapped with sret insertion.
// The POP call returns a 24-byte struct (→ sret hidden pointer at param 0)
// and takes one i32 arg with {llvm.noundef}. remapArgAttrs shifts the user
// attr to param 1 (the original arg), while param 0 (sret ptr) gets only the
// sret attribute set by the lowering.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_attrs_sret
kgen.func @arg_attrs_sret(%arg0: !kgen.scalar<si32>) {
  // CHECK: llvm.call @c_h3
  %r = pop.external_call @c_h3(%arg0)
    : (!kgen.scalar<si32>) -> !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)> <argAttrs = [{llvm.noundef}]>
  hlcf.return
}
// sret ptr at param 0, original arg with user attr at param 1.
// CHECK: llvm.func @c_h3(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, i32 {llvm.noundef})
}

// -----

// H4: resAttrs on a struct result.
// Both POP calls have resAttrs {llvm.noundef}. The struct result is coerced
// to i64, and bytes 5-7 of the struct are padding, so the declaration does
// not keep noundef. The scalar result keeps it.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @res_attrs_struct
kgen.func @res_attrs_struct() {
  // CHECK: llvm.call @c_h4_struct
  %s = pop.external_call @c_h4_struct()
    : () -> !kgen.struct<(scalar<si32>, scalar<si8>)> <resAttrs = {llvm.noundef}>
  // CHECK: llvm.call @c_h4_scalar
  %i = pop.external_call @c_h4_scalar()
    : () -> !kgen.scalar<si32> <resAttrs = {llvm.noundef}>
  hlcf.return
}
// CHECK: llvm.func @c_h4_struct() -> i64
// CHECK-NOT: noundef
// CHECK: llvm.func @c_h4_scalar() -> (i32 {llvm.noundef})
}

//===----------------------------------------------------------------------===//
// Group F: Vector type fields
//
// As in clang, a vector member is classified by its size: up to 32 bits is
// INTEGER, 64 bits is SSE whatever the element type, and 128 bits fills one
// SSE register (SSE, then SSEUP).
//===----------------------------------------------------------------------===//

// -----

// F1: {vector<2 x f32>} (8 bytes) → 64-bit vector → SSE → f64. Regression
// guard.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_vec2xf32_struct
kgen.func @arg_vec2xf32_struct(%arg0: !kgen.struct<(simd<2, f32>)>) {
  // CHECK: llvm.call @c_f1(%{{.*}}) : (f64) -> ()
  pop.external_call @c_f1(%arg0) : (!kgen.struct<(simd<2, f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_f1(f64)
}

// -----

// F2: {vector<2 x f32>, f64} (16 bytes) → SSEPair → (f64, f64).
// Eightbyte 0 (bytes 0-7): vector<2 x f32> (VectorType) → SSE → f64.
// Eightbyte 1 (bytes 8-15): f64 → SSE → f64.
// Both SSE → SSEPair. Regression guard.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_vec2xf32_and_f64
kgen.func @arg_vec2xf32_and_f64(%arg0: !kgen.struct<(simd<2, f32>, scalar<f64>)>) {
  // CHECK: llvm.call @c_f2(%{{.*}}, %{{.*}}) : (f64, f64) -> ()
  pop.external_call @c_f2(%arg0) : (!kgen.struct<(simd<2, f32>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_f2(f64, f64)
}

// -----

// F3: {vector<4 x f32>} (16 bytes) → SSE, then SSEUP → one SSE register,
// passed and returned as <2 x double>, as clang does for such a struct.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_vec4xf32_struct
kgen.func @arg_vec4xf32_struct(%arg0: !kgen.struct<(simd<4, f32>)>) {
  // CHECK: llvm.call @c_f3(%{{.*}}) : (vector<2xf64>) -> vector<2xf64>
  %r = pop.external_call @c_f3(%arg0) : (!kgen.struct<(simd<4, f32>)>) -> !kgen.struct<(simd<4, f32>)>
  hlcf.return
}
// CHECK: llvm.func @c_f3(vector<2xf64>) -> vector<2xf64>
}

// -----

// F4: {vector<2 x i32>} (8 bytes) → 64-bit vector → SSE → f64. As in clang,
// a 64-bit vector is SSE whatever its element type.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_vec2xi32_struct
kgen.func @arg_vec2xi32_struct(%arg0: !kgen.struct<(simd<2, si32>)>) {
  // CHECK: llvm.call @c_f4(%{{.*}}) : (f64) -> ()
  pop.external_call @c_f4(%arg0) : (!kgen.struct<(simd<2, si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_f4(f64)
}

//===----------------------------------------------------------------------===//
// Group N: Nested struct and array fields
//
// As in clang's X86_64ABIInfo::classify, each member of a struct without
// vector members is classified at its offset, looking inside nested structs
// and arrays, and the classes are merged per eightbyte.
// N1-N3: nested float structs → SSE. N4-N5: integer regression guards.
// N6-N8: array, empty struct and zero-length array members.
//===----------------------------------------------------------------------===//

// -----

// N1: {struct<(f32)>} (4 bytes) — CORRECT: the f32 inside the nested struct
// is SSE → 4-byte SSE struct → f32.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_float_4
kgen.func @arg_nested_float_4(%arg0: !kgen.struct<(struct<(scalar<f32>)>)>) {
  // CHECK: llvm.call @c_n1(%{{.*}}) : (f32) -> ()
  pop.external_call @c_n1(%arg0) : (!kgen.struct<(struct<(scalar<f32>)>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_n1(f32)
}

// -----

// N2: {struct<(f32,f32)>} (8 bytes) — CORRECT: both f32 inside the nested
// struct are SSE → 8-byte SSE struct → f64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_float_8
kgen.func @arg_nested_float_8(%arg0: !kgen.struct<(struct<(scalar<f32>, scalar<f32>)>)>) {
  // CHECK: llvm.call @c_n2(%{{.*}}) : (f64) -> ()
  pop.external_call @c_n2(%arg0) : (!kgen.struct<(struct<(scalar<f32>, scalar<f32>)>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_n2(f64)
}

// -----

// N3: {struct<(f32,f32)>, f64} (16 bytes) — CORRECT.
// Eightbyte 0 (bytes 0-7): the two f32 of struct<(f32,f32)> → SSE → f64.
// Eightbyte 1 (bytes 8-15): f64 → SSE → f64.
// Both SSE → SSEPair (f64, f64).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_float_and_f64
kgen.func @arg_nested_float_and_f64(%arg0: !kgen.struct<(struct<(scalar<f32>, scalar<f32>)>, scalar<f64>)>) {
  // CHECK: llvm.call @c_n3(%{{.*}}, %{{.*}}) : (f64, f64) -> ()
  pop.external_call @c_n3(%arg0) : (!kgen.struct<(struct<(scalar<f32>, scalar<f32>)>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_n3(f64, f64)
}

// -----

// N4: {struct<(i32,i32)>} (8 bytes) — CORRECT: nested struct with integer
// fields → INTEGER. Regression guard: integer nested structs must remain i64.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_int_8
kgen.func @arg_nested_int_8(%arg0: !kgen.struct<(struct<(scalar<si32>, scalar<si32>)>)>) {
  // CHECK: llvm.call @c_n4(%{{.*}}) : (i64) -> ()
  pop.external_call @c_n4(%arg0) : (!kgen.struct<(struct<(scalar<si32>, scalar<si32>)>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_n4(i64)
}

// -----

// N5: {struct<(i32,i32)>, i64} (16 bytes) — CORRECT: eightbyte 0 has
// struct<(i32,i32)> → INTEGER (correct); eightbyte 1 has i64 → INTEGER.
// IntegerPair (i64, i64). Regression guard.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_nested_int_and_i64
kgen.func @arg_nested_int_and_i64(%arg0: !kgen.struct<(struct<(scalar<si32>, scalar<si32>)>, scalar<si64>)>) {
  // CHECK: llvm.call @c_n5(%{{.*}}, %{{.*}}) : (i64, i64) -> ()
  pop.external_call @c_n5(%arg0) : (!kgen.struct<(struct<(scalar<si32>, scalar<si32>)>, scalar<si64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_n5(i64, i64)
}

// -----

// N6: {[3 x f32]} (12 bytes) — the array is classified element by element:
// eightbyte 0 holds two f32 → SSE, eightbyte 1 holds one f32 → SSE.
// SSEPair (f64, f32), the same registers as clang's (<2 x float>, float).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_array_member
kgen.func @arg_array_member(%arg0: !kgen.struct<(array<3, scalar<f32>>)>) {
  // CHECK: llvm.call @c_n6(%{{.*}}, %{{.*}}) : (f64, f32) -> ()
  pop.external_call @c_n6(%arg0) : (!kgen.struct<(array<3, scalar<f32>>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_n6(f64, f32)
}

// -----

// N7: {f32, {}, f32} (8 bytes) — as in clang, the empty struct adds no class,
// so the eightbyte is SSE → f64, the same register as clang's <2 x float>.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_empty_member
kgen.func @arg_empty_member(%arg0: !kgen.struct<(scalar<f32>, struct<()>, scalar<f32>)>) {
  // CHECK: llvm.call @c_n7(%{{.*}}) : (f64) -> ()
  pop.external_call @c_n7(%arg0) : (!kgen.struct<(scalar<f32>, struct<()>, scalar<f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_n7(f64)
}

// -----

// N8: {f32, {[0 x f32]}, f32} (8 bytes) — as in clang, the zero-length array
// adds no class, so the eightbyte is SSE → f64. The middle member is how
// InlineArray[Float32, 0] lowers. (On arm64 the same struct is not an HFA.)
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_zero_array_member
kgen.func @arg_zero_array_member(%arg0: !kgen.struct<(scalar<f32>, struct<(array<0, scalar<f32>>)>, scalar<f32>)>) {
  // CHECK: llvm.call @c_n8(%{{.*}}) : (f64) -> ()
  pop.external_call @c_n8(%arg0) : (!kgen.struct<(scalar<f32>, struct<(array<0, scalar<f32>>)>, scalar<f32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_n8(f64)
}

//===----------------------------------------------------------------------===//
// Group R: Rollback-to-stack (register exhaustion)
//
// A two-eightbyte struct needs two integer registers. When preceding integer
// args leave fewer than two free, the System V ABI passes the whole struct in
// memory (byval) rather than splitting it across a register and the stack.
//===----------------------------------------------------------------------===//

// -----

// R1: two ptrs + three i32 consume all but one integer register, so the
// trailing {i32,i32,i32} (would be IntegerPair) rolls back to memory (byval).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_rollback_after_five
kgen.func @arg_rollback_after_five(%p0: !kgen.pointer<scalar<ui8>>, %p1: !kgen.pointer<scalar<ui8>>, %a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>, %c: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_r1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr {llvm.byval = !llvm.struct<(i32, i32, i32)>}) -> ()
  pop.external_call @c_r1(%p0, %p1, %a, %b, %c, %s) : (!kgen.pointer<scalar<ui8>>, !kgen.pointer<scalar<ui8>>, !kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_r1(!llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr {llvm.byval = !llvm.struct<(i32, i32, i32)>})
}

// -----

// R2: control - one ptr + three i32 leave two integer registers free, so the
// same struct fits and stays a two-register IntegerPair (i64, i32).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_no_rollback_after_four
kgen.func @arg_no_rollback_after_four(%p0: !kgen.pointer<scalar<ui8>>, %a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>, %c: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  // CHECK: llvm.call @c_r2(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i32, i32, i64, i32) -> ()
  pop.external_call @c_r2(%p0, %a, %b, %c, %s) : (!kgen.pointer<scalar<ui8>>, !kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_r2(!llvm.ptr, i32, i32, i32, i64, i32)
}

// -----

// R3: sret control - the hidden result pointer takes one integer register;
// three i32 leave two free, so the struct still fits as IntegerPair (i64, i32).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sret_no_rollback
kgen.func @arg_sret_no_rollback(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>, %c: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  %r = pop.external_call @c_r3(%a, %b, %c, %s) : (!kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>
  hlcf.return
}
// CHECK: llvm.func @c_r3(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, i32, i32, i32, i64, i32)
}

// -----

// R4: one extra i32 over R3 tips it over the edge - sret + four i32 leave only
// one integer register, so the struct rolls back to memory (byval).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sret_rollback
kgen.func @arg_sret_rollback(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>, %c: !kgen.scalar<si32>, %d: !kgen.scalar<si32>, %s: !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) {
  %r = pop.external_call @c_r4(%a, %b, %c, %d, %s) : (!kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.scalar<si32>, !kgen.struct<(scalar<si32>, scalar<si32>, scalar<si32>)>) -> !kgen.struct<(scalar<si64>, scalar<si64>, scalar<si64>)>
  hlcf.return
}
// CHECK: llvm.func @c_r4(!llvm.ptr {llvm.sret = !llvm.struct<(i64, i64, i64)>}, i32, i32, i32, i32, !llvm.ptr {llvm.byval = !llvm.struct<(i32, i32, i32)>})
}

// -----

// R5: SSE control - six f64 leave two SSE registers free, so the {f64,f64}
// struct still fits and stays a two-register SSEPair (f64, f64).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sse_no_rollback
kgen.func @arg_sse_no_rollback(%a: !kgen.scalar<f64>, %b: !kgen.scalar<f64>, %c: !kgen.scalar<f64>, %d: !kgen.scalar<f64>, %e: !kgen.scalar<f64>, %f: !kgen.scalar<f64>, %s: !kgen.struct<(scalar<f64>, scalar<f64>)>) {
  // CHECK: llvm.call @c_r5(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, f64, f64, f64, f64, f64, f64) -> ()
  pop.external_call @c_r5(%a, %b, %c, %d, %e, %f, %s) : (!kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.struct<(scalar<f64>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_r5(f64, f64, f64, f64, f64, f64, f64, f64)
}

// -----

// R6: one extra f64 over R5 exhausts the SSE registers - seven f64 leave only
// one, so the {f64,f64} struct rolls back to memory (byval).
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_sse_rollback
kgen.func @arg_sse_rollback(%a: !kgen.scalar<f64>, %b: !kgen.scalar<f64>, %c: !kgen.scalar<f64>, %d: !kgen.scalar<f64>, %e: !kgen.scalar<f64>, %f: !kgen.scalar<f64>, %g: !kgen.scalar<f64>, %s: !kgen.struct<(scalar<f64>, scalar<f64>)>) {
  // CHECK: llvm.call @c_r6(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, f64, f64, f64, f64, f64, !llvm.ptr {llvm.byval = !llvm.struct<(f64, f64)>}) -> ()
  pop.external_call @c_r6(%a, %b, %c, %d, %e, %f, %g, %s) : (!kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.scalar<f64>, !kgen.struct<(scalar<f64>, scalar<f64>)>) -> ()
  hlcf.return
}
// CHECK: llvm.func @c_r6(f64, f64, f64, f64, f64, f64, f64, !llvm.ptr {llvm.byval = !llvm.struct<(f64, f64)>})
}

// -----

// R7: variadic args use the same registers as fixed args. One fixed and four
// variadic i64 leave one integer register, so the variadic {i64,i64} rolls
// back to memory (byval), as in clang.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_rollback
kgen.func @variadic_rollback(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %e: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si64>, scalar<si64>)>) {
  // CHECK: llvm.call @c_r7(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) vararg(!llvm.func<void (i64, ...)>) : (i64, i64, i64, i64, i64, !llvm.ptr {llvm.byval = !llvm.struct<(i64, i64)>}) -> ()
  pop.external_call @c_r7(%a, %b, %c, %d, %e, %s)
    : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si64>, scalar<si64>)>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_r7(i64, ...)
}

// -----

// R8: control - one fixed and three variadic i64 leave two integer registers,
// so the variadic {i64,i64} stays in registers.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @variadic_no_rollback
kgen.func @variadic_no_rollback(%a: !kgen.scalar<si64>, %b: !kgen.scalar<si64>, %c: !kgen.scalar<si64>, %d: !kgen.scalar<si64>, %s: !kgen.struct<(scalar<si64>, scalar<si64>)>) {
  // CHECK: llvm.call @c_r8(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) vararg(!llvm.func<void (i64, ...)>) : (i64, i64, i64, i64, i64, i64) -> ()
  pop.external_call @c_r8(%a, %b, %c, %d, %s)
    : (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.struct<(scalar<si64>, scalar<si64>)>) -> () <numFixedArgs = 1 : index>
  hlcf.return
}
// CHECK: llvm.func @c_r8(i64, ...)
}

//===----------------------------------------------------------------------===//
// Group I: Narrow integer extension
//
// Rule: clang extends a narrow integer at the producer and
// relies on it at the consumer. The sign comes from the POP dtype. Only the
// arguments are marked: a mark on the result would let LLVM assume the callee
// extended it, but a C struct flattened to a bare Mojo integer comes back with
// unspecified upper bits.
//===----------------------------------------------------------------------===//

// -----

// I1: ui8, si16, bool and si32 arguments, si8 result.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_narrow_ints
kgen.func @arg_narrow_ints(%u8: !kgen.scalar<ui8>, %s16: !kgen.scalar<si16>, %bool: !kgen.scalar<bool>, %s32: !kgen.scalar<si32>) {
  // CHECK: llvm.call @c_i1(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i8 {llvm.zeroext}, i16 {llvm.signext}, i1 {llvm.zeroext}, i32) -> i8
  %r = pop.external_call @c_i1(%u8, %s16, %bool, %s32) : (!kgen.scalar<ui8>, !kgen.scalar<si16>, !kgen.scalar<bool>, !kgen.scalar<si32>) -> !kgen.scalar<si8>
  hlcf.return
}
// CHECK: llvm.func @c_i1(i8 {llvm.zeroext}, i16 {llvm.signext}, i1 {llvm.zeroext}, i32) -> i8
}

// -----

// I2: {si8}, {ui16} and {bool} arguments, {si8} result. clang coerces a struct
// of one narrow integer to an integer of its size and does not extend it, so
// the coerced integers get no mark, whatever the sign of the field.
module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {
// CHECK-LABEL: @arg_single_field_structs
kgen.func @arg_single_field_structs(%s8: !kgen.struct<(scalar<si8>)>, %u16: !kgen.struct<(scalar<ui16>)>, %bool: !kgen.struct<(scalar<bool>)>) {
  // CHECK: llvm.call @c_i2(%{{.*}}, %{{.*}}, %{{.*}}) : (i8, i16, i8) -> i8
  %r = pop.external_call @c_i2(%s8, %u16, %bool) : (!kgen.struct<(scalar<si8>)>, !kgen.struct<(scalar<ui16>)>, !kgen.struct<(scalar<bool>)>) -> !kgen.struct<(scalar<si8>)>
  hlcf.return
}
// CHECK: llvm.func @c_i2(i8, i16, i8) -> i8
}
