// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// COM: Test that get_linkage_name with a GPU target sanitizes names:
// COM: - Adds "_" prefix to digit-starting names
// COM: - Replaces invalid characters with underscores

kgen.generator @"42kernel"() {
  hlcf.return
}

kgen.generator @"42kernel_with::special,chars"() {
  hlcf.return
}

kgen.generator @"normal_but::special"() {
  hlcf.return
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
  hlcf.return
}

// -----

// COM: Test that get_linkage_name with a GPU target sanitizes explicit linkage names:
// COM: - Adds "_" prefix to digit-starting names
// COM: - Replaces invalid characters with underscores
// COM: - Appends a hash suffix derived from the function's sym_name and type
kgen.generator @"42kernel"() attributes { linkageName = #kgen.linkage_name<"42.sanitize.this/kernel" : !kgen.string, true> } {
  hlcf.return
}

// CHECK-LABEL: kgen.func export @get_gpu_linkage_name
kgen.generator export @get_gpu_linkage_name() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda",
                                         arch = "sm_80",
                                         simd_bit_width = 128,
                                         index_bit_width = 64,
                                         tune_cpu = "sm_80",
                                         data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">>
  // CHECK: constant: string = <"_42_sanitize_this_kernel_{{[0-9a-f]+}}">
  %0 = kgen.param.constant: string = <#kgen.get_linkage_name<nvptx, #kgen.symbol.constant<@"42kernel"> : !kgen.generator<() -> ()>>>
  hlcf.return
}

// -----

// Test parametric linkage names in offload-compiled functions, with and
// without sanitization.

// A parametric linkage name concatenating the linkage name of its function
// parameter with a constant string.
kgen.generator @HELLO<x: () -> index>() -> !kgen.none
 attributes {
   linkageName = #kgen.linkage_name<
     #pop.string_concat<
       #kgen.get_linkage_name<
         current_target(),
         #kgen.param.decl.ref<"x"> : !kgen.generator<() -> index>
       >  : !kgen.string,
       "_hello"
      > : !kgen.string,
     false>
  } {
  %none = kgen.param.constant: none = <#kgen.none>
  %0 = kgen.call_param[() -> index: x]()
  hlcf.return %none : !kgen.none
}

kgen.generator @FOO() -> index
    attributes { linkageName = #kgen.linkage_name<"bar" : !kgen.string, false> } {
  %0 = index.constant 1
  hlcf.return %0 : index
}

// This linkage name needs sanitization for the nvptx target
kgen.generator @FLIP() -> index
    attributes { linkageName = #kgen.linkage_name<"1bar" : !kgen.string, false> } {
  %0 = index.constant 1
  hlcf.return %0 : index
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  kgen.param.declare *"foo()": () -> index = <@FOO>
  kgen.param.declare *"flip()": () -> index = <@FLIP>

  kgen.param.declare nvptx: target =
      <#kgen.target<triple = "nvptx64-nvidia-cuda",
                    arch = "sm_80",
                    simd_bit_width = 128,
                    index_bit_width = 64,
                    tune_cpu = "sm_80",
                    data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">
      >

  // Check that inside the offloaded module, the PTX kernel is given the right name.
  // @HELLO has mangle=false, so no hash suffix is appended. The inner
  // get_linkage_name call on "x" also produces no hash (mangle=false on @FOO),
  // so the outer name is just the concatenation of the sanitized inner name and
  // "_hello".
  // CHECK: ModuleID =
  // CHECK-SAME: define dso_local ptx_kernel void @bar_hello(
  %1 = kgen.compile_offload<
          nvptx, 2, "", "",
          :() -> !kgen.none @HELLO<:() -> index *"foo()">
       > : !kgen.struct<(string, string)>

  // Check that inside the offloaded module, the PTX kernel is given the right name.
  // @FLIP has mangle=false with a digit-leading name ("1bar"), which the
  // get_linkage_name sanitizer normalises to "_1bar". @HELLO (mangle=false)
  // concatenates that with "_hello" → "_1bar_hello", no hash suffix.
  // CHECK: ModuleID =
  // CHECK-SAME: define dso_local ptx_kernel void @_1bar_hello(
  %2 = kgen.compile_offload<
          nvptx, 2, "", "",
          :() -> !kgen.none @HELLO<:() -> index *"flip()">
       > : !kgen.struct<(string, string)>
  hlcf.return
}
