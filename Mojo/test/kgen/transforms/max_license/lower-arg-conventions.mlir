// RUN: kgen-opt -allow-unregistered-dialect -lower-arg-conventions=mem-to-reg-size-limit=256 -verify-parameters %s | FileCheck %s

#type_value = #kgen.type<typevalue<#kgen.instref<@"IntPackable">>, index> : !kgen.type
#type_value1 = #kgen.type<typevalue<#kgen.instref<@"BoolPackable">>, i1> : !kgen.type
#type_value2 = #kgen.type<pointer<typevalue<#type_value>>, pointer<index>> : !kgen.type
#type_value3 = #kgen.type<pointer<typevalue<#type_value1>>, pointer<i1>> : !kgen.type

// CHECK-LABEL: kgen.func @lower_args_with_arg_metadata
// CHECK-SAME: (%arg0: i8, %arg1: index, %arg2: i1, %arg3: index, %arg4: i1, %arg5: i16, %arg6: !kgen.struct<() isParamPack>, %arg7: i32)
kgen.func @lower_args_with_arg_metadata(
 %arg0: !kgen.pointer<i8> imm_mem,
 %arg1: !kgen.pointer<struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>> pack_imm_mem,
 %arg2: !kgen.pointer<i16> imm_mem,
 %arg3: !kgen.pointer<struct<() isParamPack>> imm_mem,
 %arg4: !kgen.pointer<i32> imm_mem
) -> !kgen.pointer<struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>> {
    hlcf.return %arg1 : !kgen.pointer<struct<(pointer<index>, pointer<i1>, #type_value2, #type_value3) isParamPack>>
} <fnArgAttrs = [{nvvm.grid_constant = unit}, {}, {nvvm.grid_constant = unit}, {}, {nvvm.grid_constant = unit}]>
// CHECK: } <{{.*}}fnArgAttrs = [{nvvm.grid_constant}, {}, {}, {}, {}, {nvvm.grid_constant}, {}, {nvvm.grid_constant}]

// Erasing the byref_result slot must shrink `fnArgAttrs` with it. Two
// survivors, distinctly keyed, so taking the wrong index cannot pass.
// CHECK-LABEL: kgen.func @byref_res_with_arg_metadata(%arg0: index owned, %arg1: index owned) -> index
kgen.func @byref_res_with_arg_metadata(
 %arg0: index owned,
 %arg1: index owned,
 %__result__: !kgen.pointer<index> byref_result
) -> !kgen.none {
  "somehow.populate"(%__result__) : (!kgen.pointer<index>) -> ()
  %none = kgen.param.constant: !kgen.none = <#kgen.none>
  hlcf.return %none : !kgen.none
} <fnArgAttrs = [{nvvm.grid_constant = unit}, {llvm.noalias = unit}, {}]>
// CHECK: } <{{.*}}fnArgAttrs = [{nvvm.grid_constant}, {llvm.noalias}]

// Same invariant on the other erasure path, the `never` error slot. It sits
// between the two survivors, so this covers erasing from the middle.
// CHECK-LABEL: kgen.func @remove_never_error_slot_with_arg_metadata
// CHECK-SAME: (%arg0: index owned, %arg1: !kgen.pointer<struct<() memoryOnly>> byref_result) throws
kgen.func @remove_never_error_slot_with_arg_metadata(
 %live: index owned,
 %err: !kgen.pointer<!kgen.never> byref_error,
 %arg0: !kgen.pointer<struct<() memoryOnly>> byref_result
) throws -> !kgen.scalar<bool> {
  %1 = kgen.param.constant: scalar<bool> = <false>
  hlcf.return %1 : !kgen.scalar<bool>
} <fnArgAttrs = [{llvm.noalias = unit}, {}, {nvvm.grid_constant = unit}]>
// CHECK: } <{{.*}}fnArgAttrs = [{llvm.noalias}, {nvvm.grid_constant}]
