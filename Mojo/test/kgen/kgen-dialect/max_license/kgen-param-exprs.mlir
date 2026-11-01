// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

kgen.generator @kernel() {
  hlcf.return
}

// CHECK-LABEL: kgen.generator @target_is_gpu_triple<t0: target>()
kgen.generator @target_is_gpu_triple<t0: target>() {
  kgen.param.assert <identical(:string target_get_field(t0, "triple"), "nvptx64-nvidia-cuda")>, "triple must be nvptx64-nvidia-cuda"
  hlcf.return
}

// A GPU triple reports its own architecture in `triple_arch`, so the host
// predicates built on that field are False there rather than accidentally
// matching.
// CHECK-LABEL: kgen.generator @target_get_triple_arch_gpu()
kgen.generator @target_get_triple_arch_gpu() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_100a", features = "", data_layout = "", simd_bit_width = 128>>
  kgen.param.assert<identical(:string target_get_field(nvptx, "triple_arch"), "nvptx64")>,
                    "triple arch is nvptx64"
  hlcf.return
}

// CHECK-LABEL: @compile_assembly
kgen.generator @compile_assembly<emission_kind: index>() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>>
  kgen.param.declare amd: target = <#kgen.target<triple = "amdgcn-amd-amdhsa", arch = "", data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", simd_bit_width = 128>>

  // CHECK: constant: string = <#kgen.compile_assembly<nvptx, =asm, "", false, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<nvptx, =asm, "", false, :() -> () @kernel>>
  // CHECK: constant: string = <#kgen.compile_assembly<nvptx, =llvm, "", true, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<nvptx, =llvm, "", true, :() -> () @kernel>>
  // CHECK: constant: string = <#kgen.compile_assembly<nvptx, =llvm, "option1=value1,option2=value2", true, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<nvptx, =llvm, "option1=value1,option2=value2", true, :() -> () @kernel>>

  // CHECK: constant: string = <#kgen.compile_assembly<amd, =object, "", true, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<amd, =object, "", true, :() -> () @kernel>>
  // CHECK: constant: string = <#kgen.compile_assembly<nvptx, emission_kind, "", true, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<nvptx, emission_kind, "", true, :() -> () @kernel>>

  hlcf.return
}
