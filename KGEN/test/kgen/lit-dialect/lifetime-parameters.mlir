// RUN: kgen-opt %s -verify-parameters -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode     -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @lifetimes
lit.func @lifetimes() {
  // CHECK: partial: !lit.signature<[1]<index>(!lit.ref<index, mut *[0,0]>) -> ()>
  lit.alias.decl partial: !lit.signature<[1]<index>(!lit.ref<index, mut *[0,0]>) -> ()> = <?>
  lit.var.decl "x" var : !lit.ref<index, mut *"a`">
  kgen.return
}

// CHECK-LABEL: @decls
// CHECK-SAME: <x: dtype, y>[imm a, mut b](%ptr: !lit.ref<simd<y, x>, mut b>)
lit.func @decls<x: dtype, y>[imm a, mut b](%ptr: !lit.ref<simd<y, x>, mut b>) {
  // CHECK: ref: !lit.signature<[2]<"x": dtype, "y": index>("ptr": !lit.ref<simd<*(0,1), *(0,0)>, mut *[0,1]>) -> ()> = <@decls>
  lit.alias.decl ref: !lit.signature<[2]<"x": dtype, "y": index>("ptr": !lit.ref<simd<*(0,1), *(0,0)>, mut *[0,1]>) -> ()> = <@decls>
  kgen.return
}

lit.func @callee[mut a](%out: !lit.ref<index, mut a>) -> !lit.ref<index, mut a> {
  kgen.return %out : !lit.ref<index, mut a>
}

lit.func @async_callee[mut a](%out: !lit.ref<index, mut a>) async -> !lit.ref<index, mut a> {
  kgen.return %out : !lit.ref<index, mut a>
}

lit.func @calls(%f: !lit.signature<[1](!lit.ref<index, mut *[0,0]>) -> !lit.ref<index, mut *[0,0]>>) {
  %x = lit.var.decl "x" var : !lit.ref<index, mut a>

  // CHECK: lit.call @callee[mut a](%x) : !lit.signature<[1]("out": !lit.ref<index, mut *[0,0]>) -> !lit.ref<index, mut *[0,0]>>
  %0 = lit.call @callee[mut a](%x) : !lit.signature<[1]("out": !lit.ref<index, mut *[0,0]>) -> !lit.ref<index, mut *[0,0]>>
  // CHECK: lit.call_indirect %f[mut a](%x) : !lit.signature<[1](!lit.ref<index, mut *[0,0]>) -> !lit.ref<index, mut *[0,0]>>
  %1 = lit.call_indirect %f[mut a](%x) : !lit.signature<[1](!lit.ref<index, mut *[0,0]>) -> !lit.ref<index, mut *[0,0]>>
  // CHECK: = lit.async.call[!lit.signature<[1]("out": !lit.ref<index, mut *[0,0]>) async -> !lit.ref<index, mut *[0,0]>>: @async_callee][mut a](%x)
  %2 = lit.async.call[!lit.signature<[1]("out": !lit.ref<index, mut *[0,0]>) async -> !lit.ref<index, mut *[0,0]>>: @async_callee][mut a](%x)

  // COM: Anchor the types to ensure they match.
  // CHECK: "use"
  // CHECK-COUNT-2: !lit.ref<index, mut a>
  // CHECK: !co.routine
  "use"(%0, %1, %2) : (!lit.ref<index, mut a>, !lit.ref<index, mut a>, !co.routine) -> ()

  kgen.return
}
