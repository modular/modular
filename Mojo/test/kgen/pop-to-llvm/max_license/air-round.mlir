// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

// On Apple Metal/AIR, llvm.roundeven (and llvm.nearbyint) crash
// MTLCompilerService when JIT-ing the GPU machine code, so pop.round must lower
// to llvm.rint on a Metal target instead. llvm.rint rounds using the current FP
// rounding mode (round-to-nearest-even by default), matching pop.round's
// banker's-rounding semantics, so the result is numerically identical. The
// non-Metal llvm.roundeven path is covered by @round_simd in simd.mlir.

module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK-LABEL: @round_metal
kgen.func @round_metal(%arg0: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.call_intrinsic "llvm.rint"
  // CHECK-NOT: llvm.call_intrinsic "llvm.roundeven"
  %0 = pop.round %arg0 : !kgen.simd<4, f32>
  hlcf.return %0 : !kgen.simd<4, f32>
}

}

// -----

// Same lowering on metal:2 / apple-m1 (the cross-compile the macOS CI runs):
// the choice keys on the triple, so the metal triple selects llvm.rint
// regardless of the AIR/Metal version generation.
module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m1",
    data_layout = "">} {

// CHECK-LABEL: @round_metal_m1
kgen.func @round_metal_m1(%arg0: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
  // CHECK: llvm.call_intrinsic "llvm.rint"
  // CHECK-NOT: llvm.call_intrinsic "llvm.roundeven"
  %0 = pop.round %arg0 : !kgen.simd<4, f32>
  hlcf.return %0 : !kgen.simd<4, f32>
}

}
