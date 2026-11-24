// Which fp8 types a GPU converts natively depends on the architecture, and a
// cast lowered on the architecture that implements the other encoding is off
// by a binade, because the OCP and `fnuz` types differ in exponent bias. A
// pair the target does not implement goes to the precompiled snippet instead.
//
// RUN: kgen-opt -split-input-file \
// RUN:   -pass-pipeline='builtin.module(kgen.func(legalize-pop-operations),lower-global-pop-to-llvm{inline-precompiled-mlir=false},kgen.func(lower-pop-to-llvm{inline-precompiled-mlir=false}))' \
// RUN:   %s | FileCheck %s

// gfx942 implements the `fnuz` pair.
// CHECK-LABEL: @gfx942_fnuz
// CHECK: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"
module attributes {M.target_info = #M.target<triple = "amdgpu9.42-amd-amdhsa", arch="gfx942", data_layout="">} {
  kgen.func @gfx942_fnuz(%x: !kgen.scalar<f32>) -> !kgen.scalar<f8e4m3fnuz> {
    %0 = pop.cast %x : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fnuz>
    hlcf.return %0 : !kgen.scalar<f8e4m3fnuz>
  }
}

// -----

// The same chip does not implement the OCP pair, which shares the mnemonic.
// CHECK-LABEL: @gfx942_ocp
// CHECK: llvm.call @__kgen_pop_cast_f32_to_f8e4m3fn
// CHECK-NOT: llvm.amdgcn.cvt.pk.fp8.f32
module attributes {M.target_info = #M.target<triple = "amdgpu9.42-amd-amdhsa", arch="gfx942", data_layout="">} {
  kgen.func @gfx942_ocp(%x: !kgen.scalar<f32>) -> !kgen.scalar<f8e4m3fn> {
    %0 = pop.cast %x : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }
}

// -----

// CDNA4 implements the OCP pair.
// CHECK-LABEL: @gfx950_ocp
// CHECK: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"
module attributes {M.target_info = #M.target<triple = "amdgpu9.50-amd-amdhsa", arch="gfx950", data_layout="">} {
  kgen.func @gfx950_ocp(%x: !kgen.scalar<f32>) -> !kgen.scalar<f8e4m3fn> {
    %0 = pop.cast %x : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }
}

// -----

// ... and not the `fnuz` pair.
// CHECK-LABEL: @gfx950_fnuz
// CHECK: llvm.call @__kgen_pop_cast_f32_to_f8e4m3fnuz
// CHECK-NOT: llvm.amdgcn.cvt.pk.fp8.f32
module attributes {M.target_info = #M.target<triple = "amdgpu9.50-amd-amdhsa", arch="gfx950", data_layout="">} {
  kgen.func @gfx950_fnuz(%x: !kgen.scalar<f32>) -> !kgen.scalar<f8e4m3fnuz> {
    %0 = pop.cast %x : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fnuz>
    hlcf.return %0 : !kgen.scalar<f8e4m3fnuz>
  }
}

// -----

// Hopper converts f32 to the OCP types with `cvt`.
// CHECK-LABEL: @sm_90_ocp
// CHECK: cvt.rn.satfinite.e4m3x2.f32
module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch="sm_90a", data_layout="">} {
  kgen.func @sm_90_ocp(%x: !kgen.scalar<f32>) -> !kgen.scalar<f8e4m3fn> {
    %0 = pop.cast %x : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }
}

// -----

// sm_120 reaches them through software, so the snippet answers instead.
// CHECK-LABEL: @sm_120_ocp
// CHECK: llvm.call @__kgen_pop_cast_f32_to_f8e4m3fn
// CHECK-NOT: cvt.rn.satfinite
module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch="sm_120a", data_layout="">} {
  kgen.func @sm_120_ocp(%x: !kgen.scalar<f32>) -> !kgen.scalar<f8e4m3fn> {
    %0 = pop.cast %x : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }
}
