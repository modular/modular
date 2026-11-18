// Tests the `inline-precompiled-mlir` option that selects whether an fp8 -> f32
// `pop.cast` is lowered to a call to the precompiled conversion snippet, or the
// snippet is marked for LLVM to inline. The empty (host) target has no native
// fp8 conversion, so the software snippet path is used.
//
// It also pins which of the two snippets a cast reaches: a scalar cast takes
// the one that kept the conversion's branches, a SIMD cast the flat one its
// lanes can be re-vectorized from.
//
// Call path: `lower-global-pop-to-llvm` inserts the snippet function and
// `lower-pop-to-llvm` emits an `llvm.call` to it.
// RUN: kgen-opt -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(legalize-pop-operations),lower-global-pop-to-llvm{inline-precompiled-mlir=false},llvm.func(lower-pop-to-llvm{inline-precompiled-mlir=false}),llvm.func(lower-control-flow),llvm.func(reconcile-unrealized-casts))' %s \
// RUN:   | FileCheck %s --check-prefix=CALL
//
// Inline path: the same call, but the snippet is marked `always_inline` for
// LLVM to inline. Splicing it here would split the caller's block, which
// structured control flow forbids.
// RUN: kgen-opt -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(legalize-pop-operations),lower-global-pop-to-llvm{inline-precompiled-mlir=true},llvm.func(lower-pop-to-llvm{inline-precompiled-mlir=true}),llvm.func(lower-control-flow),llvm.func(reconcile-unrealized-casts))' %s \
// RUN:   | FileCheck %s --check-prefix=INLINE

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CALL-LABEL: llvm.func internal @cvt
// CALL: llvm.call @__kgen_pop_cast_f8e4m3fn_to_f32_branching(%arg0)
// CALL: llvm.return

// INLINE-LABEL: llvm.func internal @cvt
// INLINE: llvm.call @__kgen_pop_cast_f8e4m3fn_to_f32_branching(%arg0)
kgen.func @cvt(%x: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f32> {
  %0 = pop.cast %x : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f32>
  hlcf.return %0 : !kgen.scalar<f32>
}

// The snippets are scalar, so a SIMD cast is one conversion per lane. The
// trailing `(%` keeps these from matching the `_branching` symbol.
// CALL-LABEL: llvm.func internal @cvt4
// CALL-COUNT-4: llvm.call @__kgen_pop_cast_f8e4m3fn_to_f32(%
// CALL: llvm.return
// Each snippet the module uses is inserted once at module scope.
// CALL: llvm.func internal @__kgen_pop_cast_f8e4m3fn_to_f32_branching(
// CALL: llvm.func internal @__kgen_pop_cast_f8e4m3fn_to_f32(

// INLINE-LABEL: llvm.func internal @cvt4
// INLINE-COUNT-4: llvm.call @__kgen_pop_cast_f8e4m3fn_to_f32(%
kgen.func @cvt4(%x: !kgen.simd<4, f8e4m3fn>) -> !kgen.simd<4, f32> {
  %0 = pop.cast %x : !kgen.simd<4, f8e4m3fn> to !kgen.simd<4, f32>
  hlcf.return %0 : !kgen.simd<4, f32>
}

}

// INLINE: llvm.func internal @__kgen_pop_cast_f8e4m3fn_to_f32_branching(
// INLINE-SAME: always_inline
// INLINE: llvm.func internal @__kgen_pop_cast_f8e4m3fn_to_f32(
// INLINE-SAME: always_inline
