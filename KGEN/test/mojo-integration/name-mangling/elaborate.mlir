// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// COM: Test that get_linkage_name with a GPU target sanitizes names:
// COM: - Adds "_" prefix to digit-starting names
// COM: - Encodes non-alnum characters

kgen.generator @"42kernel"() {
  kgen.return
}

kgen.generator @"42kernel_with::special,chars"() {
  kgen.return
}

kgen.generator @"normal_but::special"() {
  kgen.return
}

// CHECK-LABEL: kgen.func export @get_gpu_linkage_name
kgen.generator export @get_gpu_linkage_name() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda",
                                         arch = "sm_80",
                                         simd_bit_width = 128,
                                         index_bit_width = 64,
                                         tune_cpu = "sm_80",
                                         data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">>
  // CHECK: constant: string = <"_42kernel">
  %0 = kgen.param.constant: string = <#kgen.get_linkage_name<nvptx, #kgen.symbol.constant<@"42kernel"> : !kgen.generator<() -> ()>>>
  // CHECK: constant: string = <"_42kernel_with_special_chars6A6AsA">
  %1 = kgen.param.constant: string = <#kgen.get_linkage_name<nvptx, #kgen.symbol.constant<@"42kernel_with::special,chars"> : !kgen.generator<() -> ()>>>
  // CHECK: constant: string = <"normal_but_special6A6A">
  %2 = kgen.param.constant: string = <#kgen.get_linkage_name<nvptx, #kgen.symbol.constant<@"normal_but::special"> : !kgen.generator<() -> ()>>>
  kgen.return
}

// -----

// COM: Test that get_linkage_name with a GPU target sanitizes explicit linkage names:
// COM: - Adds "_" prefix to digit-starting names
// COM: - Encodes non-alnum characters
kgen.generator @"42kernel"() attributes { linkageName = "42.sanitize.this/kernel" : !kgen.string } {
  kgen.return
}

// CHECK-LABEL: kgen.func export @get_gpu_linkage_name
kgen.generator export @get_gpu_linkage_name() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda",
                                         arch = "sm_80",
                                         simd_bit_width = 128,
                                         index_bit_width = 64,
                                         tune_cpu = "sm_80",
                                         data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">>
  // CHECK: constant: string = <"_42_sanitize_this_kerneluAuAvA">
  %0 = kgen.param.constant: string = <#kgen.get_linkage_name<nvptx, #kgen.symbol.constant<@"42kernel"> : !kgen.generator<() -> ()>>>
  kgen.return
}

// -----

kgen.generator @some_generator() attributes {sourceName = "foo"} {
  kgen.return
}

// CHECK-LABEL: kgen.func export @get_source_name
kgen.generator export @get_source_name() {
  // CHECK-NEXT: constant: string = <"foo">
  %0 = kgen.param.constant: string = <#kgen.get_source_name<#kgen.symbol.constant<@some_generator> : !kgen.generator<() -> ()>>>
  kgen.return
}
