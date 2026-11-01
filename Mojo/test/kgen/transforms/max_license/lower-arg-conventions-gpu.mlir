// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-arg-conventions -verify-parameters %s | FileCheck %s

// -----

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch="">} {

  // CHECK-LABEL: promote_to_imm_mem
  // CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>)>> imm_mem
  kgen.func export @promote_to_imm_mem(%arg0: !kgen.struct<(pointer<none>)>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    // CHECK-NEXT: pop.load %arg0 : !kgen.pointer<struct<(pointer<none>)>>
    %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<none>)>
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }

  // CHECK-LABEL: kgen.func @fn
  // CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>)>> imm_mem
  kgen.func @fn(%arg0: !kgen.struct<(pointer<none>)>) -> !kgen.none {
    // CHECK-NEXT: pop.load %arg0 : !kgen.pointer<struct<(pointer<none>)>>
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }

  // CHECK-LABEL: promote_to_imm_mem_call
  // CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>)>> imm_mem
  kgen.func export @promote_to_imm_mem_call(%arg0: !kgen.struct<(pointer<none>)>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    // CHECK-NEXT: %[[P0:.*]] = pop.load %arg0 : !kgen.pointer<struct<(pointer<none>)>>
    // CHECK-NEXT: %[[P1:.*]] = pop.stack_allocation 1 x struct<(pointer<none>)>
    // CHECK-NEXT: pop.store %[[P0]], %[[P1]] : !kgen.pointer<struct<(pointer<none>)>>
    // CHECK-NEXT: kgen.call @fn(%[[P1]]) : (!kgen.pointer<struct<(pointer<none>)>> imm_mem) -> !kgen.none
    %0 = kgen.call @fn(%arg0) : (!kgen.struct<(pointer<none>)>) -> !kgen.none
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }

  // CHECK-LABEL: dont_promote_to_imm_mem
  // CHECK-SAME: (%arg0: !kgen.struct<(scalar<f32>)>
  kgen.func export @dont_promote_to_imm_mem(%arg0: !kgen.struct<(scalar<f32>)>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "amdgcn-amd-amdhsa", arch="">} {

  // CHECK-LABEL: promote_to_imm_mem
  // CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>)>> imm_mem
  kgen.func export @promote_to_imm_mem(%arg0: !kgen.struct<(pointer<none>)>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    // CHECK-NEXT: pop.load %arg0 : !kgen.pointer<struct<(pointer<none>)>>
    %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<none>)>
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }

  // CHECK-LABEL: kgen.func @fn
  // CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>)>> imm_mem
  kgen.func @fn(%arg0: !kgen.struct<(pointer<none>)>) -> !kgen.none {
    // CHECK-NEXT: pop.load %arg0 : !kgen.pointer<struct<(pointer<none>)>>
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }

  // CHECK-LABEL: promote_to_imm_mem_call
  // CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>)>> imm_mem
  kgen.func export @promote_to_imm_mem_call(%arg0: !kgen.struct<(pointer<none>)>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    // CHECK-NEXT: %[[P0:.*]] = pop.load %arg0 : !kgen.pointer<struct<(pointer<none>)>>
    // CHECK-NEXT: %[[P1:.*]] = pop.stack_allocation 1 x struct<(pointer<none>)>
    // CHECK-NEXT: pop.store %[[P0]], %[[P1]] : !kgen.pointer<struct<(pointer<none>)>>
    // CHECK-NEXT: kgen.call @fn(%[[P1]]) : (!kgen.pointer<struct<(pointer<none>)>> imm_mem) -> !kgen.none
    %0 = kgen.call @fn(%arg0) : (!kgen.struct<(pointer<none>)>) -> !kgen.none
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }

  // CHECK-LABEL: dont_promote_to_imm_mem
  // CHECK-SAME: (%arg0: !kgen.struct<(scalar<f32>)>
  kgen.func export @dont_promote_to_imm_mem(%arg0: !kgen.struct<(scalar<f32>)>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }
}

// -----

// COM: Packs + GPU (TODO: MOCO-1885)
module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch="">} {

  // CHECK-LABEL: promote_to_imm_mem
  // CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>)>> imm_mem
  kgen.func export @promote_to_imm_mem(%arg0: !kgen.struct<(struct<(pointer<none>)>) isParamPack>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    // CHECK-NEXT: pop.load %arg0 : !kgen.pointer<struct<(pointer<none>)>>
    // CHECK-NEXT: kgen.struct.create
    %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(struct<(pointer<none>)>) isParamPack>
    %1 = kgen.struct.extract %0[0] : !kgen.struct<(pointer<none>)>
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }

  // CHECK-LABEL: promote_to_imm_mem_call
  // CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>)>> imm_mem
  kgen.func export @promote_to_imm_mem_call(%arg0: !kgen.struct<(struct<(pointer<none>)>) isParamPack>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    // CHECK: pop.load %arg0 : !kgen.pointer<struct<(pointer<none>)>>
    // CHECK-NEXT: kgen.struct.create
    // CHECK-NEXT: kgen.struct.extract {{.*}} : <(struct<(pointer<none>)>) isParamPack>
    // CHECK-NEXT: pop.stack_allocation 1 x struct<(pointer<none>)>
    // CHECK-NEXT: pop.store {{.*}} : !kgen.pointer<struct<(pointer<none>)>>
    %0 = kgen.call @promote_to_imm_mem(%arg0) : (!kgen.struct<(struct<(pointer<none>)>) isParamPack>) -> !kgen.none
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch="">} {

  // CHECK-LABEL: add_extra_indirection
  // CHECK-SAME: (%arg0: !kgen.pointer<pointer<none>>
  kgen.func export @add_extra_indirection(%arg0: !kgen.pointer<none>) -> !kgen.none attributes {fnAttrs = {kgen.offload.kernelid = 1 : index}} {
    // CHECK: %[[LOAD:.*]] = pop.load %arg0 : !kgen.pointer<pointer<none>>
    // CHECK-NEXT: pop.load %[[LOAD]] : !kgen.pointer<none>
    %0 = pop.load %arg0 : !kgen.pointer<none>
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }

  // CHECK-LABEL: pack_with_pointers
  // CHECK-SAME: (%arg0: !kgen.pointer<pointer<none>> imm_mem,
  // CHECK-SAME: %arg1: !kgen.pointer<pointer<none>> imm_mem,
  // CHECK-SAME: %arg2: !kgen.scalar<f32>
  kgen.func @pack_with_pointers(%arg0: !kgen.pointer<!kgen.struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<scalar<f32>>) isParamPack>> imm_mem) -> !kgen.none always_inline {
    // CHECK-NEXT: pop.stack_allocation 1 x scalar<f32>
    // CHECK-NEXT: pop.store {{.*}} : !kgen.pointer<scalar<f32>>
    // CHECK: pop.load {{.*}} : !kgen.pointer<pointer<none>>
    // CHECK: pop.load {{.*}} : !kgen.pointer<pointer<none>>
    // CHECK: pop.load {{.*}} : !kgen.pointer<scalar<f32>>
    %0 = pop.load %arg0 : !kgen.pointer<!kgen.struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<scalar<f32>>) isParamPack>>
    %1 = kgen.struct.extract %0[0] : <(pointer<pointer<none>>, pointer<pointer<none>>, pointer<scalar<f32>>) isParamPack>
    %2 = kgen.struct.extract %0[1] : <(pointer<pointer<none>>, pointer<pointer<none>>, pointer<scalar<f32>>) isParamPack>
    %3 = kgen.struct.extract %0[2] : <(pointer<pointer<none>>, pointer<pointer<none>>, pointer<scalar<f32>>) isParamPack>
    %5 = pop.load %1 : !kgen.pointer<pointer<none>>
    %6 = pop.load %2 : !kgen.pointer<pointer<none>>
    %7 = pop.load %3 : !kgen.pointer<scalar<f32>>
    %none = kgen.param.constant: none = <#kgen.none>
    hlcf.return %none : !kgen.none
  }
}
