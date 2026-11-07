// Platforms on `register(name, cond)`. The kernels are in
// GraphCompiler/test/kernels/platform_*_stubs.

// The import keeps each platform as its canonical `#kgen.gen`, and gives a
// registration for all devices no platform.
// The import loads `builtin_kernels`, so grep for the decls under test.
// RUN: mo-opt %s --split-input-file --mogg-import-packages="mojo-pkg-paths=%platform_pkg" 2>&1 | grep 'registration = "pred_' | FileCheck %s --check-prefix=IMPORT

// The test pass prints each platform and its value on the devices of
// `mogg.test_device_infos`, compares the platforms of each registration, and
// resolves each registration on the devices of `mo.device_info_mapping`.
// RUN: mo-opt %s --split-input-file --mogg-import-packages="mojo-pkg-paths=%platform_pkg" --mogg-test-kernel-registrations -o /dev/null 2>&1 | FileCheck %s

// A decl with a platform prints in a form that parses back.
// RUN: mo-opt %s --split-input-file --mogg-import-packages="mojo-pkg-paths=%platform_pkg" | mo-opt --split-input-file --mogg-test-kernel-registrations -o /dev/null 2>&1 | FileCheck %s --check-prefix=ROUNDTRIP

// The import rejects each forged platform of a package, and keeps scanning
// after the first.
// RUN: not mo-opt %s --split-input-file --mlir-print-op-on-diagnostic=false --mogg-import-packages="mojo-pkg-paths=%platform_invalid_pkg" -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID

// The import rejects two registrations of an op for the same devices, however
// they are written.
// RUN: not mo-opt %s --split-input-file --mlir-print-op-on-diagnostic=false --mogg-import-packages="mojo-pkg-paths=%platform_duplicate_pkg" -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUPLICATE

// The import rejects a platform that holds for no device.
// RUN: not mo-opt %s --split-input-file --mlir-print-op-on-diagnostic=false --mogg-import-packages="mojo-pkg-paths=%platform_unsatisfiable_pkg" -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSATISFIABLE

// Registry validation finds the ambiguous op without a device.
// RUN: not mo-opt %s --split-input-file --mlir-print-op-on-diagnostic=false --mogg-import-packages="mojo-pkg-paths=%platform_pkg" --mogg-validate-kernel-registry -o /dev/null 2>&1 | FileCheck %s --check-prefix=VALIDATE

// IMPORT: @{{.*}}.PredMatmulCpu registration = "pred_matmul" platform_predicate = #kgen.gen<identical(:string *(0,0), "cpu")> : !kgen.generator<{{.*}}> execute
// IMPORT: @{{.*}}.PredMixedAllDevices registration = "pred_mixed" execute

// CHECK: platform @{{.*}}.PredB200 = type("gpu") & api("cuda") & arch("sm_100a") & model("NVIDIA B200")
// CHECK: platform @{{.*}}.PredB200 on #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200"> = true
// CHECK: platform @{{.*}}.PredB200 on #M.device_info<"gpu", "cuda", "sm_103a", "NVIDIA B300"> = false

// `ACCELERATOR` holds for every accelerator, the TPU too.
// CHECK: platform @{{.*}}.PredMatmulAccelerator = type("gpu")
// CHECK: platform @{{.*}}.PredMatmulAccelerator on #M.device_info<"gpu", "tpu", "tpu-v6e", "unknown"> = true
// CHECK: platform @{{.*}}.PredMatmulCpu = type("cpu")
// CHECK: platform @{{.*}}.PredMatmulHip = type("gpu") & api("hip")
// CHECK: platform @{{.*}}.PredMatmulTcgen05 = (arch("sm_100a") | arch("sm_101a") | arch("sm_103a")) & type("gpu") & api("cuda")
// CHECK: platform @{{.*}}.PredMatmulTcgen05 on #M.device_info<"gpu", "cuda", "sm_103a", "NVIDIA B300"> = true
// CHECK: platform @{{.*}}.PredMatmulTcgen05 on #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100"> = false
// CHECK: platform @{{.*}}.PredMatmulTpu = type("gpu") & api("tpu")
// CHECK: platform @{{.*}}.PredMatmulTpuV6e = type("gpu") & api("tpu") & arch("tpu-v6e")
// CHECK: compare @{{.*}}.PredMatmulAccelerator @{{.*}}.PredMatmulCpu = incomparable
// CHECK: compare @{{.*}}.PredMatmulAccelerator @{{.*}}.PredMatmulHip = second narrower
// CHECK: compare @{{.*}}.PredMatmulAccelerator @{{.*}}.PredMatmulTcgen05 = second narrower
// CHECK: compare @{{.*}}.PredMatmulAccelerator @{{.*}}.PredMatmulTpu = second narrower
// CHECK: compare @{{.*}}.PredMatmulHip @{{.*}}.PredMatmulTcgen05 = incomparable
// CHECK: compare @{{.*}}.PredMatmulTcgen05 @{{.*}}.PredMatmulTpu = incomparable
// CHECK: compare @{{.*}}.PredMatmulTpu @{{.*}}.PredMatmulTpuV6e = second narrower

// A registration without a platform holds for all devices.
// CHECK: platform @{{.*}}.PredMixedAllDevices = true
// CHECK: compare @{{.*}}.PredMixedAllDevices @{{.*}}.PredMixedTcgen05 = second narrower

// CHECK: platform @{{.*}}.PredMyAccelerators = ((type("gpu") & api("cuda")) | (type("gpu") & api("hip")) | (type("gpu") & api("tpu"))) & type("gpu")
// CHECK: platform @{{.*}}.PredMyAccelerators on #M.device_info<"gpu", "hip", "gfx942", "AMD Instinct MI300X"> = true
// CHECK: platform @{{.*}}.PredMyAccelerators on #M.device_info<"gpu", "tpu", "tpu-v6e", "unknown"> = true
// CHECK: compare @{{.*}}.PredOverlapBlackwell @{{.*}}.PredOverlapHopperBlackwell = incomparable
// CHECK: compare @{{.*}}.PredSpecializedBlackwell @{{.*}}.PredSpecializedHopperBlackwell = incomparable
// CHECK: compare @{{.*}}.PredSpecializedBlackwell @{{.*}}.PredSpecializedSm100 = second narrower
// CHECK: compare @{{.*}}.PredSpecializedHopperBlackwell @{{.*}}.PredSpecializedSm100 = second narrower

// The first module has a CPU and a B200.
// CHECK: resolve "pred_b200" on "cpu" -> error: no kernel matching device info for 'pred_b200' on device #M.device_info<"cpu", "cpu", "unknown", "unknown">; candidates:
// CHECK-NEXT: - {{.*}}.PredB200: cond=type("gpu") & api("cuda") & arch("sm_100a") & model("NVIDIA B200")
// CHECK: resolve "pred_b200" on "gpu" -> @{{.*}}.PredB200
// CHECK: resolve "pred_matmul" on "cpu" -> @{{.*}}.PredMatmulCpu
// CHECK: resolve "pred_matmul" on "gpu" -> @{{.*}}.PredMatmulTcgen05
// CHECK: resolve "pred_mixed" on "gpu" -> @{{.*}}.PredMixedTcgen05
// CHECK: resolve "pred_my_accelerators" on "gpu" -> @{{.*}}.PredMyAccelerators
// CHECK: resolve "pred_overlap" on "gpu" -> error: ambiguous kernel resolution for 'pred_overlap' on device #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200">: user-defined registrations match, and none of them applies to a strict subset of the devices of all the others:
// CHECK-NEXT: {{.*}}platform_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: note: {{.*}}.PredOverlapBlackwell: cond=(arch("sm_100a") | arch("sm_103a")) & type("gpu") & api("cuda")
// CHECK-NEXT: {{.*}}platform_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: note: {{.*}}.PredOverlapHopperBlackwell: cond=(arch("sm_100a") | arch("sm_90a")) & type("gpu") & api("cuda")
// CHECK-NEXT: {{.*}}.PredOverlapBlackwell and {{.*}}.PredOverlapHopperBlackwell: only the first applies to {type="gpu", api="cuda", arch="sm_103a", model=*}, and only the second applies to {type="gpu", api="cuda", arch="sm_90a", model=*}; both apply to {type="gpu", api="cuda", arch="sm_100a", model=*}
// CHECK-NEXT: narrow a cond, or register a kernel for exactly the devices where they overlap
// CHECK: resolve "pred_specialized" on "gpu" -> @{{.*}}.PredSpecializedSm100

// The second module has an MI300X.
// CHECK: resolve "pred_matmul" on "gpu" -> @{{.*}}.PredMatmulHip
// CHECK: resolve "pred_mixed" on "gpu" -> @{{.*}}.PredMixedAllDevices

// The third module has an H100: only the broad kernels match.
// CHECK: resolve "pred_matmul" on "gpu" -> @{{.*}}.PredMatmulAccelerator
// CHECK: resolve "pred_overlap" on "gpu" -> @{{.*}}.PredOverlapHopperBlackwell
// CHECK: resolve "pred_specialized" on "gpu" -> @{{.*}}.PredSpecializedHopperBlackwell

// The fourth module has a TPU v6e: the TPU v6e kernel is narrower than the
// TPU and accelerator kernels, and the kernel for all devices matches it.
// CHECK: resolve "pred_b200" on "gpu" -> error: no kernel matching device info for 'pred_b200' on device #M.device_info<"gpu", "tpu", "tpu-v6e", "unknown">
// CHECK: resolve "pred_matmul" on "gpu" -> @{{.*}}.PredMatmulTpuV6e
// CHECK: resolve "pred_mixed" on "gpu" -> @{{.*}}.PredMixedAllDevices
// CHECK: resolve "pred_my_accelerators" on "gpu" -> @{{.*}}.PredMyAccelerators

// The fifth module has a TPU v5e: the TPU kernel is the narrowest match.
// CHECK: resolve "pred_matmul" on "gpu" -> @{{.*}}.PredMatmulTpu

// ROUNDTRIP: platform @{{.*}}.PredMatmulTcgen05 = (arch("sm_100a") | arch("sm_101a") | arch("sm_103a")) & type("gpu") & api("cuda")
// ROUNDTRIP: resolve "pred_matmul" on "gpu" -> @{{.*}}.PredMatmulTcgen05
// ROUNDTRIP: resolve "pred_matmul" on "gpu" -> @{{.*}}.PredMatmulHip

// INVALID: platform_invalid_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: invalid platform: a platform must compare each device field with a string literal, not #kgen.param.identical<#kgen.param.index.ref<0, 0> : !kgen.string, #kgen.param.index.ref<0, 1> : !kgen.string>
// INVALID: platform_invalid_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: invalid platform: unsupported expression in platform: {{.*}}@StringLiteral
// INVALID: platform_invalid_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: invalid platform: unsupported expression in platform: {{.*}}startswith
// INVALID: platform_invalid_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: invalid platform: a platform must take the label, api, arch, and model of a device as strings and return a bool, not !kgen.generator<!lit.generator<<"label": !lit.struct<@std::@simd::@SIMD
// INVALID: error: failed to scan kernel package

// DUPLICATE-DAG: platform_duplicate_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: kernel 'pred_dup_reordered' is registered multiple times for the same devices
// DUPLICATE-DAG: platform_duplicate_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: kernel 'pred_dup_distributed' is registered multiple times for the same devices
// DUPLICATE-DAG: platform_duplicate_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: kernel 'pred_dup_all_devices' is registered multiple times for the same devices
// DUPLICATE-DAG: platform_duplicate_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: note: the other registration is here

// UNSATISFIABLE: platform_unsatisfiable_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: invalid platform: it folds to `false`, so it holds for no device

// VALIDATE-NOT: 'pred_matmul'
// VALIDATE: error: ambiguous kernel resolution for 'pred_overlap' on device {type="gpu", api="cuda", arch="sm_100a", model=*}: user-defined registrations match
// VALIDATE-NOT: 'pred_specialized'

module attributes {
  mo.device_info_mapping = {
    cpu = #M.device_info<"cpu", "cpu", "unknown", "unknown">,
    gpu = #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200">
  },
  mogg.test_device_infos = [
    #M.device_info<"cpu", "cpu", "unknown", "unknown">,
    #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200">,
    #M.device_info<"gpu", "cuda", "sm_103a", "NVIDIA B300">,
    #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100">,
    #M.device_info<"gpu", "hip", "gfx942", "AMD Instinct MI300X">,
    #M.device_info<"gpu", "tpu", "tpu-v6e", "unknown">
  ]
} {
}

// -----

module attributes {
  mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "hip", "gfx942", "AMD Instinct MI300X">
  }
} {
}

// -----

module attributes {
  mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "cuda", "sm_90a", "NVIDIA H100">
  }
} {
}

// -----

module attributes {
  mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "tpu", "tpu-v6e", "unknown">
  }
} {
}

// -----

module attributes {
  mo.device_info_mapping = {
    gpu = #M.device_info<"gpu", "tpu", "tpu-v5e", "unknown">
  }
} {
}
