// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck --check-prefixes=CHECK %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck --check-prefixes=CHECK %s

// CHECK-LABEL: kgen.func @"metadata,x=16,y=16,z=8"
// CHECK-SAME: fnArgAttrs = [{}, {nvvm.maxntid = #pop.array<16, 16, 8>
// CHECK-SAME: fnAttrs = {nvvm.maxntid = #pop.array<16, 16, 8>
kgen.generator @metadata<x: i32, y: i32, z: i32>(%a: i32, %b: i32) attributes {
  fnArgAttrs = [
    [],
    ["nvvm.maxntid",  #pop.array<x, y, z> : !pop.array<3, i32>]
  ],
  fnAttrs = [
    "nvvm.maxntid",  #pop.array<x, y, z> : !pop.array<3, i32>
  ]
}{
  hlcf.return
}

kgen.generator @kernel(%a: i32, %b: i32) {
  kgen.call @metadata<:i32 16, :i32 16, :i32 8>(%a, %b) : (i32, i32) -> ()
  hlcf.return
}

// -----

// COM: A compile_offload op over a function HELLO lowers to a struct of the
// compiled assembly and the module name.
kgen.generator @HELLO<x: () -> index>() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  %0 = kgen.call_param[() -> index: x]()
  hlcf.return %none : !kgen.none
}

kgen.generator @FOO() -> index {
  %0 = index.constant 1
  hlcf.return %0 : index
}

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  kgen.param.declare *"foo()": () -> index = <@FOO>
  // CHECK: [[STR:%.*]] = kgen.param.constant: string = <"; Mod{{.*}}">
  // CHECK-NEXT: [[STR1:%.*]] = kgen.param.constant: string = <"{{.*}}">
  // CHECK-NEXT: kgen.struct.create([[STR]], [[STR1]]) : !kgen.struct<(string, string)>
  // CHECK-NOT: populate_captures

  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda",
                                         arch = "sm_80",
                                         simd_bit_width = 128,
                                         index_bit_width = 64,
                                         tune_cpu = "sm_80",
                                         data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">>
  %1 = kgen.compile_offload<nvptx, 2, "", "",
                            :() -> !kgen.none @HELLO<:() -> index *"foo()">>
                            : !kgen.struct<(string, string)>
  hlcf.return
}

// -----

// The offload target-attribute dictionary is forwarded from the generator
// onto the instantiated function, where the target backend reads it.

// CHECK-LABEL: kgen.func @target_attrs_forwarding
// CHECK-SAME: kgen.target_attrs = {tpu.grid_dims = [2, 4]}
kgen.generator @target_attrs_forwarding() -> index attributes {"kgen.target_attrs" = {tpu.grid_dims = [2, 4]}} {
  %c = kgen.param.constant = <42>
  hlcf.return %c : index
}
kgen.generator @main() -> index {
  %v = kgen.call @target_attrs_forwarding() : () -> index
  hlcf.return %v : index
}
