// RUN: kgen-opt %s -lower-lit -allow-unregistered-dialect | kgen-opt -verify-parameters -kgen-print-inline-type-values | FileCheck %s

// `lit.memcpy` requires its element types to be canonically equal, not
// identical, so a value typed by a sugared alias can be copied into a slot typed
// by the alias' expansion without a rebind in between.

lit.struct.decl @Wrapper<T: type> {
  lit.struct.field data : !kgen.param<T>
}

// Only the expanded form of the sugar has to be right; the sugared form is a
// stand-in for the alias that would normally spell it.
!Sugared = !lit.struct<@Wrapper<:type sugar_alias(f32, si32)>>
!Canonical = !lit.struct<@Wrapper<:type si32>>

// CHECK-LABEL: kgen.generator @memcpy_canonically_equal
lit.fn @memcpy_canonically_equal<l: !lit.origin<true>, l2: !lit.origin<true>>
    (%src: !lit.ref<!Sugared, mut l>, %dst: !lit.ref<!Canonical, mut l2>) {
  // CHECK: [[LEN:%.*]] = kgen.param.constant = <get_sizeof(
  // CHECK-NEXT: pop.memcpy %arg1, %arg0, [[LEN]]
  lit.memcpy %src, %dst : !lit.ref<!Sugared, mut l> -> !lit.ref<!Canonical, mut l2>
  hlcf.return
}
