// Before M5, Metal has no AIR fp8 converts, so fp8 casts go to the
// precompiled snippets.
//
// RUN: kgen-opt -split-input-file \
// RUN:   -pass-pipeline='builtin.module(kgen.func(legalize-pop-operations),lower-global-pop-to-llvm{inline-precompiled-mlir=false},kgen.func(lower-pop-to-llvm{inline-precompiled-mlir=false}))' \
// RUN:   %s | FileCheck %s
// RUN: kgen-opt -split-input-file \
// RUN:   -pass-pipeline='builtin.module(kgen.func(legalize-pop-operations),lower-global-pop-to-llvm{inline-precompiled-mlir=false},kgen.func(lower-pop-to-llvm{inline-precompiled-mlir=false}))' \
// RUN:   %s | FileCheck %s --check-prefix=NO-AIR

// NO-AIR-NOT: air.convert

module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m3",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK-LABEL: @test_f8e5m2_to_f32
// CHECK: llvm.call @__kgen_pop_cast_f8e5m2_to_f32
kgen.func @test_f8e5m2_to_f32(%x: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f32> {
  %r = pop.cast %x : !kgen.scalar<f8e5m2> to !kgen.scalar<f32>
  hlcf.return %r : !kgen.scalar<f32>
}

// CHECK-LABEL: @test_f32_to_f8e5m2
// CHECK: llvm.call @__kgen_pop_cast_f32_to_f8e5m2
kgen.func @test_f32_to_f8e5m2(%x: !kgen.scalar<f32>) -> !kgen.scalar<f8e5m2> {
  %r = pop.cast %x : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
  hlcf.return %r : !kgen.scalar<f8e5m2>
}

// f16 has no snippet of its own, so the cast widens through f32 first.
// CHECK-LABEL: @test_f8e4m3fn_to_f16
// CHECK: llvm.call @__kgen_pop_cast_f8e4m3fn_to_f32
// CHECK: llvm.fptrunc
kgen.func @test_f8e4m3fn_to_f16(%x: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f16> {
  %r = pop.cast %x : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
  hlcf.return %r : !kgen.scalar<f16>
}

// A SIMD cast, which Metal otherwise sends to `air.convert` whole, is split
// into one snippet call per lane.
// CHECK-LABEL: @test_v4f32_to_f8e4m3fn
// CHECK-COUNT-4: llvm.call @__kgen_pop_cast_f32_to_f8e4m3fn
kgen.func @test_v4f32_to_f8e4m3fn(%x: !kgen.simd<4, f32>) -> !kgen.simd<4, f8e4m3fn> {
  %r = pop.cast %x : !kgen.simd<4, f32> to !kgen.simd<4, f8e4m3fn>
  hlcf.return %r : !kgen.simd<4, f8e4m3fn>
}

}
