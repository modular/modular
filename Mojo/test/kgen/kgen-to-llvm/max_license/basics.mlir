// RUN: kgen-opt -lower-kgen-to-llvm -split-input-file %s | FileCheck %s

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
// CHECK-LABEL: llvm.func @kernel() attributes {nvvm.kernel, dso_local
kgen.func export @kernel() {
  hlcf.return
}
}

// -----

module attributes {M.target_info = #M.target<triple = "amdgpu9.42-amd-amdhsa", arch = "gfx942", data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", simd_bit_width = 128>} {
// CHECK-LABEL: llvm.func amdgpu_kernelcc @kernel() attributes {dso_local,
kgen.func export @kernel() {
  hlcf.return
}
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {

// CHECK-LABEL: llvm.func @llvm_metadata
// CHECK-SAME: nvvm.intval = 4 : i64
// CHECK-SAME: nvvm.maxntid = array<i32: 256, 1, 4>
// CHECK-SAME: nvvm.unitattr
// CHECK-SAME: passthrough = [{{.*}}, ["intval", "2"], ["strval", "hello"], "unitattr"]
kgen.func export @llvm_metadata() attributes {
  fnAttrs = {
    llvm.unitattr,
    llvm.intval = 2,
    llvm.strval = "hello",

    nvvm.unitattr,
    nvvm.intval = 4,
    nvvm.maxntid = #pop.array<256, 1, 4> : !pop.array<3, i32>
  }
} {
  hlcf.return
}

}

// -----

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_90", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
// CHECK-LABEL: llvm.func @kgen_fp8_param_constant
kgen.func export @kgen_fp8_param_constant() {
  // CHECK: llvm.mlir.constant(56 : i8) : i8
  %0 = kgen.param.constant: f8E4M3FN = <1.>

  // CHECK: llvm.mlir.constant(60 : i8) : i8
  %1 = kgen.param.constant: f8E5M2 = <1.>
  hlcf.return
}
}

// -----

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_90", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {

// CHECK-LABEL: kgen_fp8_args
// CHECK-SAME: %[[ARG0:.+]]: vector<16xi8>
kgen.func @kgen_fp8_args(%arg0: !kgen.simd<16,f8e5m2>) -> !kgen.simd<16,f8e5m2> {
  // CHECK: llvm.return %[[ARG0]] : vector<16xi8>
  hlcf.return %arg0: !kgen.simd<16,f8e5m2>
}
}

// -----

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_90", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {

// CHECK-LABEL: kgen_fp8_ptr_arg
kgen.func @kgen_fp8_ptr_arg(%arg0: !kgen.pointer<scalar<f8e5m2>>) -> !kgen.scalar<f8e5m2>{
  %0 = pop.load %arg0 : !kgen.pointer<scalar<f8e5m2>>
  // CHECK: llvm.return %2 : i8
  hlcf.return %0 : !kgen.scalar<f8e5m2>
}

}

// -----

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_90", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {

// CHECK-LABEL: kgen_fp4_ptr_arg
kgen.func @kgen_fp4_ptr_arg(%arg0: !kgen.pointer<scalar<f4e2m1fn>>) -> !kgen.scalar<f4e2m1fn>{
  %0 = pop.load %arg0 : !kgen.pointer<scalar<f4e2m1fn>>
  // CHECK: llvm.return %2 : i4
  hlcf.return %0 : !kgen.scalar<f4e2m1fn>
}

// CHECK-LABEL: kgen_fp6_e2m3_ptr_arg
kgen.func @kgen_fp6_e2m3_ptr_arg(%arg0: !kgen.pointer<scalar<f6e2m3fn>>) -> !kgen.scalar<f6e2m3fn>{
  %0 = pop.load %arg0 : !kgen.pointer<scalar<f6e2m3fn>>
  // CHECK: llvm.return %2 : i6
  hlcf.return %0 : !kgen.scalar<f6e2m3fn>
}

// CHECK-LABEL: kgen_fp6_e3m2_ptr_arg
kgen.func @kgen_fp6_e3m2_ptr_arg(%arg0: !kgen.pointer<scalar<f6e3m2fn>>) -> !kgen.scalar<f6e3m2fn>{
  %0 = pop.load %arg0 : !kgen.pointer<scalar<f6e3m2fn>>
  // CHECK: llvm.return %2 : i6
  hlcf.return %0 : !kgen.scalar<f6e3m2fn>
}

}

// -----

module attributes {M.target_info = #M.target<triple = "amdgpu9.42-amd-amdhsa", arch = "gfx942", data_layout = "", simd_bit_width = 128, index_bit_width = 64>} {
  // CHECK: llvm.func amdgpu_kernelcc @test() attributes {{{.*}}rocdl.flat_work_group_size = "1, 128"{{.*}}}
  kgen.func export @test() attributes {fnAttrs = {rocdl.flat_work_group_size = #pop.array<128> : !pop.array<1, scalar<si32>>}} {
    hlcf.return
  }

  // CHECK: llvm.func amdgpu_kernelcc @test1() attributes {{{.*}}rocdl.flat_work_group_size = "2, 128"{{.*}}}
  kgen.func export @test1() attributes {fnAttrs = {rocdl.flat_work_group_size = #pop.array<2, 128> : !pop.array<2, scalar<si32>>}} {
    hlcf.return
  }
}
