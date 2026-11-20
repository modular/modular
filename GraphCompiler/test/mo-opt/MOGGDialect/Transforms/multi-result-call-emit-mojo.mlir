// RUN: mo-opt %s --mogg-emit-mojo --mogg-dump-mojo 2>&1 | FileCheck %s

// A MOGG-ABI stub may return several values. They cross as one Mojo tuple and
// the call site destructures it, so each result stays a single SSA value.

// CHECK: def producer(arg__0: OpaquePointer[MutAnyOrigin]) raises -> Tuple[OpaquePointer[MutAnyOrigin], OpaquePointer[MutAnyOrigin]]:
// CHECK-NEXT: return (arg__0, arg__0)
mogg.func @producer private (%a: !mogg.pointer) -> (!mogg.pointer, !mogg.pointer) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"producer">>} {
  mogg.output %a, %a : !mogg.pointer, !mogg.pointer
}

// CHECK: def consumer(arg__0: OpaquePointer[MutAnyOrigin]) raises -> OpaquePointer[MutAnyOrigin]:
// CHECK: var [[PAIR:var[_0-9]+]] = producer(arg__0)
// CHECK: var [[FIRST:var[_0-9]+]] = [[PAIR]][0]
// CHECK: var [[SECOND:var[_0-9]+]] = [[PAIR]][1]
// CHECK: return [[SECOND]]
mogg.func @consumer private (%a: !mogg.pointer) -> !mogg.pointer attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"consumer">>} {
  %p:2 = mogg.call @producer %a : !mogg.pointer -> !mogg.pointer, !mogg.pointer
  mogg.output %p#1 : !mogg.pointer
}
