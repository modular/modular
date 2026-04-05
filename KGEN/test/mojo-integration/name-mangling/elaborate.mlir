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

// -----

// Test parametric linkage names in offload-compiled functions, with and
// without sanitization.

// A parametric linkage name concatenating the linkage name of its function
// parameter with a constant string.
kgen.generator @HELLO<x: () capturing -> index>() capturing -> !kgen.none
 attributes {
   linkageName = #pop.string_concat<
     #kgen.get_linkage_name<
       current_target(),
       #kgen.param.decl.ref<"x"> : !kgen.generator<() capturing -> index>
     >  : !kgen.string,
     "_hello"
    > : !kgen.string
  } {
  %none = kgen.param.constant: none = <#kgen.none>
  %0 = kgen.call_param[() capturing -> index: x]()
  kgen.return %none : !kgen.none
}

kgen.generator @FOO() capturing -> index
    attributes { linkageName = "bar" : !kgen.string } {
  %0 = pop.compiler.global_load "CAPTURE_0" : !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// This linkage name needs sanitization for the nvptx target
kgen.generator @FLIP() capturing -> index
    attributes { linkageName = "1bar" : !kgen.string } {
  %0 = pop.compiler.global_load "CAPTURE_0" : !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry(%arg0: !kgen.pointer<none>) {
  %0 = pop.stack_allocation 1 x index marked
  pop.compiler.global_store "CAPTURE_0", %0 : !kgen.pointer<index>
  kgen.param.declare *"foo()": () capturing -> index = <@FOO>
  kgen.param.declare *"flip()": () capturing -> index = <@FLIP>

  kgen.param.declare nvptx: target =
      <#kgen.target<triple = "nvptx64-nvidia-cuda",
                    arch = "sm_80",
                    simd_bit_width = 128,
                    index_bit_width = 64,
                    tune_cpu = "sm_80",
                    data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">
      >

  // Check that inside the offloaded module, the PTX kernel is given the right name.
  // CHECK: ModuleID =
  // CHECK-SAME: define dso_local ptx_kernel void @bar_hello(
  %1 = kgen.compile_offload<
          nvptx, 2, "", "",
          :() capturing -> !kgen.none @HELLO<:() capturing -> index *"foo()">
       > : !kgen.struct<(string, index)>
  kgen.param.declare x: (!kgen.pointer<none>) capturing -> !kgen.none
      = <#kgen.compile_offload_closure<
          nvptx,
          #kgen.symbol.constant<
            @HELLO<:() capturing -> index *"foo()">
          > : !kgen.generator<() capturing -> !kgen.none>>
        >
  // Check that the linkage names are also present in the populate_captures function
  // CHECK: kgen.call @bar_hello_populate_captures(%arg0)
  // CHECK-SAME: : (!kgen.pointer<none>) capturing -> !kgen.none
  %2 = kgen.call_param[(!kgen.pointer<none>) capturing -> !kgen.none: x](%arg0)

  // Check that inside the offloaded module, the PTX kernel is given the right name.
  // CHECK: ModuleID =
  // CHECK-SAME: define dso_local ptx_kernel void @_1bar_hello(
  %3 = kgen.compile_offload<
          nvptx, 2, "", "",
          :() capturing -> !kgen.none @HELLO<:() capturing -> index *"flip()">
       > : !kgen.struct<(string, index)>
  kgen.param.declare y: (!kgen.pointer<none>) capturing -> !kgen.none
      = <#kgen.compile_offload_closure<
          nvptx,
          #kgen.symbol.constant<
            @HELLO<:() capturing -> index *"flip()">
          > : !kgen.generator<() capturing -> !kgen.none>>
        >
  // CHECK: kgen.call @_1bar_hello_populate_captures(%arg0)
  // CHECK-SAME: : (!kgen.pointer<none>) capturing -> !kgen.none
  %4 = kgen.call_param[(!kgen.pointer<none>) capturing -> !kgen.none: y](%arg0)
  kgen.return
}
