// RUN: kgen-opt %s -verify-parameters | FileCheck %s

// CHECK-LABEL: @lifetimes
lit.func @lifetimes() {
  // CHECK: partial: !lit.signature<[1]<index>(!lit.ref<mut index, *[0,0]>) -> ()>
  lit.alias.decl partial: !lit.signature<[1]<index>(!lit.ref<mut index, *[0,0]>) -> ()> = <?>
  lit.varlet.decl "x" var : !lit.ref<mut index, *"`a">
  kgen.return
}

// CHECK-LABEL: @decls
// CHECK-SAME: [a, b]<x: dtype, y>(%ptr[ptr]: !lit.ref<mut simd<y, x>, b>)
lit.func @decls[a, b]<x: dtype, y>(%ptr[ptr]: !lit.ref<mut simd<y, x>, b>) {
  // CHECK: ref: !lit.signature<[2]<dtype, index>("ptr": !lit.ref<mut simd<*(0,1), *(0,0)>, *[0,1]>) -> ()> = <@decls>
  lit.alias.decl ref: !lit.signature<[2]<dtype, index>("ptr": !lit.ref<mut simd<*(0,1), *(0,0)>, *[0,1]>) -> ()> = <@decls>
  kgen.return
}
