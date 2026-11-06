// RUN: mo-opt %s --mogg-test-kernel-registrations -o /dev/null 2>&1 | FileCheck %s

// The pass reads each platform in canonical form and evaluates it on the
// devices of `mogg.test_device_infos`.
// CHECK: platform @Cpu = type("cpu")
// CHECK: platform @Cpu on #M.device_info<"cpu", "cpu", "unknown", "unknown"> = true
// CHECK: platform @Cpu on #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200"> = false
// CHECK: platform @Cuda = type("gpu") & api("cuda")
// CHECK: platform @Cuda on #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100"> = true
// CHECK: platform @Cuda on #M.device_info<"gpu", "hip", "gfx942", "AMD Instinct MI300X"> = false
// CHECK: platform @CudaReordered = type("gpu") & api("cuda")
// CHECK: platform @Blackwell = (arch("sm_100a") | arch("sm_103a")) & type("gpu") & api("cuda")
// CHECK: platform @Blackwell on #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200"> = true
// CHECK: platform @Blackwell on #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100"> = false
// CHECK: platform @B200 = type("gpu") & api("cuda") & arch("sm_100a") & model("NVIDIA B200")

// Comparison is by set of devices, not by syntax.
// CHECK: compare @Cpu @Accelerator = incomparable
// CHECK: compare @Accelerator @Cuda = second narrower
// CHECK: compare @Cuda @CudaReordered = equivalent
// CHECK: compare @Blackwell @HopperBlackwell = incomparable
// CHECK: compare @Blackwell @B200 = second narrower

// CHECK: platform @FieldToField = error: a platform must compare each device field with a string literal, not #kgen.param.identical<#kgen.param.index.ref<0, 0> : !kgen.string, #kgen.param.index.ref<0, 1> : !kgen.string>

// A decl without a platform is not printed.
// CHECK-NOT: @NoPlatform

!pred = !kgen.generator<!lit.generator<<"label": string, "api": string, "arch": string, "model": string>scalar<bool>>>

module attributes {
  mogg.test_device_infos = [
    #M.device_info<"cpu", "cpu", "unknown", "unknown">,
    #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200">,
    #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100">,
    #M.device_info<"gpu", "hip", "gfx942", "AMD Instinct MI300X">
  ]
} {
  mogg.kernel.decl @Cpu registration = "op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "cpu")> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @Accelerator registration = "op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "gpu")> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @Cuda registration = "op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @CudaReordered registration = "op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(identical(:string *(0,1), "cuda"), identical(:string *(0,0), "gpu"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @Blackwell registration = "op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_100a"), identical(:string *(0,2), "sm_103a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @HopperBlackwell registration = "op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_90a"), identical(:string *(0,2), "sm_100a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @B200 registration = "op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"), identical(:string *(0,2), "sm_100a"), identical(:string *(0,3), "NVIDIA B200"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @FieldToField registration = "invalid_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), *(0,1))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @NoPlatform registration = "no_platform_op" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
}
