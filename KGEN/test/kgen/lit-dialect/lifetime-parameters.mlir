// RUN: kgen-opt %s -verify-parameters -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode     -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @lifetimes
lit.func @lifetimes() {
  // CHECK: partial: !lit.signature<[1]<index>(!lit.ref<mut index, *[0,0]>) -> ()>
  lit.alias.decl partial: !lit.signature<[1]<index>(!lit.ref<mut index, *[0,0]>) -> ()> = <?>
  lit.varlet.decl "x" var : !lit.ref<mut index, *"`a">
  kgen.return
}

// CHECK-LABEL: @decls
// CHECK-SAME: [a, b]<x: dtype, y>(%ptr: !lit.ref<mut simd<y, x>, b>)
lit.func @decls[a, b]<x: dtype, y>(%ptr: !lit.ref<mut simd<y, x>, b>) {
  // CHECK: ref: !lit.signature<[2]<dtype, index>("ptr": !lit.ref<mut simd<*(0,1), *(0,0)>, *[0,1]>) -> ()> = <@decls>
  lit.alias.decl ref: !lit.signature<[2]<dtype, index>("ptr": !lit.ref<mut simd<*(0,1), *(0,0)>, *[0,1]>) -> ()> = <@decls>
  kgen.return
}

lit.func @callee[a](%out: !lit.ref<mut index, a>) -> !lit.ref<mut index, a> {
  kgen.return %out : !lit.ref<mut index, a>
}

lit.func @async_callee[a](%out: !lit.ref<mut index, a>) async -> !lit.ref<mut index, a> {
  kgen.return %out : !lit.ref<mut index, a>
}

lit.func @calls(%f: !lit.signature<[1](!lit.ref<mut index, *[0,0]>) -> !lit.ref<mut index, *[0,0]>>) {
  %x = lit.varlet.decl "x" var : !lit.ref<mut index, a>

  // CHECK: lit.call @callee[a](%x) : !lit.signature<[1]("out": !lit.ref<mut index, *[0,0]>) -> !lit.ref<mut index, *[0,0]>>
  %0 = lit.call @callee[a](%x) : !lit.signature<[1]("out": !lit.ref<mut index, *[0,0]>) -> !lit.ref<mut index, *[0,0]>>
  // CHECK: lit.call_param[!lit.signature<[1]("out": !lit.ref<mut index, *[0,0]>) -> !lit.ref<mut index, *[0,0]>>: @callee][a](%x)
  %1 = lit.call_param[!lit.signature<[1]("out": !lit.ref<mut index, *[0,0]>) -> !lit.ref<mut index, *[0,0]>>: @callee][a](%x)
  // CHECK: lit.call_signature %f[a](%x) : !lit.signature<[1](!lit.ref<mut index, *[0,0]>) -> !lit.ref<mut index, *[0,0]>>
  %2 = lit.call_signature %f[a](%x) : !lit.signature<[1](!lit.ref<mut index, *[0,0]>) -> !lit.ref<mut index, *[0,0]>>
  // CHECK: = lit.async.call[!lit.signature<[1]("out": !lit.ref<mut index, *[0,0]>) async -> !lit.ref<mut index, *[0,0]>>: @async_callee][a](%x)
  %3 = lit.async.call[!lit.signature<[1]("out": !lit.ref<mut index, *[0,0]>) async -> !lit.ref<mut index, *[0,0]>>: @async_callee][a](%x)

  // COM: Anchor the types to ensure they match.
  // CHECK: "use"
  // CHECK-COUNT-3: !lit.ref<mut index, a>
  // CHECK: !pop.coroutine<() -> !lit.ref<mut index, a>>
  "use"(%0, %1, %2, %3) : (!lit.ref<mut index, a>, !lit.ref<mut index, a>, !lit.ref<mut index, a>, !pop.coroutine<() -> !lit.ref<mut index, a>>) -> ()

  kgen.return
}
