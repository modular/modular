// Checks that a `denormal-fp-math-f32=<mode>` emission option of a comptime
// `#kgen.compile_assembly` sets the f32 denormal mode of the NVPTX kernel. The
// emission kind 2 is `llvm-opt`.
// RUN: kgen %s -elaborate -S -o - | FileCheck %s

kgen.generator @kernel() {
  hlcf.return
}

// CHECK-LABEL: kgen.func export @with_mode
// CHECK: constant: string
// CHECK-SAME: denormal_fpenv(float: preservesign)
kgen.generator export @with_mode() -> !kgen.string {
  kgen.param.declare nvptx: target = <#kgen.target<
    triple = "nvptx64-nvidia-cuda",
    arch = "sm_80",
    data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",
    simd_bit_width = 128
  >>
  %0 = kgen.param.constant: string = <#kgen.struct.extract<#kgen.compile_assembly<nvptx, 2, "denormal-fp-math-f32=preserve-sign", false, :() -> () @kernel>, 0>>
  hlcf.return %0 : !kgen.string
}

// CHECK-LABEL: kgen.func export @without_mode
// CHECK: constant: string
// CHECK-NOT: denormal_fpenv
// CHECK: hlcf.return
kgen.generator export @without_mode() -> !kgen.string {
  kgen.param.declare nvptx: target = <#kgen.target<
    triple = "nvptx64-nvidia-cuda",
    arch = "sm_80",
    data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",
    simd_bit_width = 128
  >>
  %0 = kgen.param.constant: string = <#kgen.struct.extract<#kgen.compile_assembly<nvptx, 2, "", false, :() -> () @kernel>, 0>>
  hlcf.return %0 : !kgen.string
}
