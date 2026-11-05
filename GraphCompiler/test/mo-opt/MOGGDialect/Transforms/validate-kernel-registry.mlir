// RUN: not mo-opt %s --mlir-print-op-on-diagnostic=false --mogg-validate-kernel-registry -o /dev/null 2>&1 | FileCheck %s
// RUN: not mo-opt %s --mlir-print-op-on-diagnostic=false --mogg-validate-kernel-registry="builtins-only=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=BUILTINS

// Registrations for disjoint or nested sets of devices are not ambiguous.
// CHECK-NOT: 'disjoint_op'
// CHECK-NOT: 'nested_op'

// An invalid platform does not stop the scan: the ambiguities found before
// and after it are all reported, with the error. The pass emits the error
// before any ambiguity, wherever the bad op falls in the scan.
// CHECK: error: invalid platform of 'InvalidPlatform': a platform must compare each device field with a string literal, not #kgen.param.identical<#kgen.param.index.ref<0, 0> : !kgen.string, #kgen.param.index.ref<0, 1> : !kgen.string>{{.*}}

// The ambiguity before the invalid platform is reported too.
// CHECK: error: ambiguous kernel resolution for 'aa_overlap_op' on device {type="gpu", api="cuda", arch="sm_100a", model=*}: built-in registrations match

// Two registrations that each cover a device the other does not tie on the
// devices they share.
// CHECK: error: ambiguous kernel resolution for 'overlap_op' on device {type="gpu", api="cuda", arch="sm_100a", model=*}: built-in registrations match, and none of them applies to a strict subset of the devices of all the others:
// Each candidate is a note at its registration.
// CHECK: note: Blackwell: cond=(arch("sm_100a") | arch("sm_103a")) & type("gpu") & api("cuda")
// CHECK: note: HopperBlackwell: cond=(arch("sm_100a") | arch("sm_90a")) & type("gpu") & api("cuda")
// CHECK: Blackwell and HopperBlackwell: only the first applies to {type="gpu", api="cuda", arch="sm_103a", model=*}, and only the second applies to {type="gpu", api="cuda", arch="sm_90a", model=*}; both apply to {type="gpu", api="cuda", arch="sm_100a", model=*}

// Registrations for the same set of devices tie wherever they apply.
// CHECK: error: ambiguous kernel resolution for 'same_op' on device {type="gpu", api="cuda", arch=*, model=*}: built-in registrations match
// CHECK: Cuda and CudaReordered apply to the same devices

// A user-defined registration shadows the built-in ones, so it hides the tie.
// CHECK-NOT: 'shadowed_op'
// BUILTINS: error: ambiguous kernel resolution for 'shadowed_op' on device

// A registration for more devices than the tied ones is not part of the tie.
// CHECK: error: ambiguous kernel resolution for 'triple_broad_op' on device {type="gpu", api="cuda", arch="sm_100a", model=*}: built-in registrations match
// CHECK: note: TripleBroadA: cond=
// CHECK: note: TripleBroadB: cond=
// CHECK-NOT: TripleBroadAll
// CHECK: TripleBroadA and TripleBroadB: only the first applies to

// Three registrations that each cover a device the others do not all tie, and
// every pair of them is compared.
// CHECK: error: ambiguous kernel resolution for 'triple_op' on device {type="gpu", api="cuda", arch="sm_100a", model=*}: built-in registrations match
// CHECK: note: TripleA: cond=
// CHECK: note: TripleB: cond=
// CHECK: note: TripleC: cond=
// CHECK: TripleA and TripleB: only the first applies to {{.*}}; both apply to {type="gpu", api="cuda", arch="sm_100a", model=*}
// CHECK: TripleA and TripleC: only the first applies to {{.*}}; both apply to {type="gpu", api="cuda", arch="sm_100a", model=*}
// CHECK: TripleB and TripleC: only the first applies to {{.*}}; both apply to {type="gpu", api="cuda", arch="sm_100a", model=*}

// The ambiguity after the invalid platform is reported too.
// CHECK: error: ambiguous kernel resolution for 'zz_overlap_op' on device {type="gpu", api="cuda", arch="sm_100a", model=*}: built-in registrations match
!pred = !kgen.generator<!lit.generator<<"label": string, "api": string, "arch": string, "model": string>scalar<bool>>>

module attributes {
  mo.device_info_mapping = {
    cpu = #M.device_info<"cpu", "cpu", "unknown", "unknown">,
    gpu = #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200">
  }
} {

  mogg.kernel.decl @AaBlackwell registration = "aa_overlap_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_100a"), identical(:string *(0,2), "sm_103a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @AaHopperBlackwell registration = "aa_overlap_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_90a"), identical(:string *(0,2), "sm_100a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @Cpu registration = "disjoint_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "cpu")> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @Accelerator registration = "disjoint_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "gpu")> : !pred struct_params = [] is_user_defined = false

  mogg.kernel.decl @NestedAccelerator registration = "nested_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "gpu")> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @NestedCuda registration = "nested_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false

  mogg.kernel.decl @Blackwell registration = "overlap_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_100a"), identical(:string *(0,2), "sm_103a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @HopperBlackwell registration = "overlap_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_90a"), identical(:string *(0,2), "sm_100a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false

  mogg.kernel.decl @Cuda registration = "same_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @CudaReordered registration = "same_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(identical(:string *(0,1), "cuda"), identical(:string *(0,0), "gpu"))> : !pred struct_params = [] is_user_defined = false

  mogg.kernel.decl @ShadowedBlackwell registration = "shadowed_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_100a"), identical(:string *(0,2), "sm_103a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @ShadowedHopperBlackwell registration = "shadowed_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_90a"), identical(:string *(0,2), "sm_100a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @UserAccelerator registration = "shadowed_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "gpu")> : !pred struct_params = [] is_user_defined = true

  mogg.kernel.decl @TripleBroadA registration = "triple_broad_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_100a"), identical(:string *(0,2), "sm_103a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @TripleBroadB registration = "triple_broad_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_90a"), identical(:string *(0,2), "sm_100a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @TripleBroadAll registration = "triple_broad_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "gpu")> : !pred struct_params = [] is_user_defined = false

  mogg.kernel.decl @TripleA registration = "triple_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_100a"), identical(:string *(0,2), "sm_103a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @TripleB registration = "triple_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_90a"), identical(:string *(0,2), "sm_100a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @TripleC registration = "triple_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_80"), identical(:string *(0,2), "sm_100a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false

  mogg.kernel.decl @TripleOkAccelerator registration = "triple_ok" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "gpu")> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @TripleOkCuda registration = "triple_ok" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @TripleOkBlackwell registration = "triple_ok" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"), identical(:string *(0,2), "sm_100a"))> : !pred struct_params = [] is_user_defined = false

  mogg.kernel.decl @InvalidPlatform registration = "invalid_platform_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), *(0,1))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @InvalidPlatformOther registration = "invalid_platform_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<identical(:string *(0,0), "gpu")> : !pred struct_params = [] is_user_defined = false

  mogg.kernel.decl @ZzBlackwell registration = "zz_overlap_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_100a"), identical(:string *(0,2), "sm_103a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
  mogg.kernel.decl @ZzHopperBlackwell registration = "zz_overlap_op" device_info = <"", "", "", ""> platform_predicate = #kgen.gen<and(or(identical(:string *(0,2), "sm_90a"), identical(:string *(0,2), "sm_100a")), identical(:string *(0,0), "gpu"), identical(:string *(0,1), "cuda"))> : !pred struct_params = [] is_user_defined = false
}
