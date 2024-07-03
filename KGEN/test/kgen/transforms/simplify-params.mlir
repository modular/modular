// RUN: kgen-opt %s -split-input-file -allow-unregistered-dialect -verify-parameters='simplify=true enable-interp=true' -verify-parameters | FileCheck %s

kgen.generator @unbound_fn<p0>() {
  kgen.return
}

kgen.generator @callee_simd<size, dt: dtype, value: simd<size, dt>>() {
  kgen.return
}

kgen.generator @callee_fn<f: () -> ()>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @param_prop
kgen.generator @param_prop<p0, p1>() {
  // CHECK-NOT: declare a0 = <p0>
  kgen.param.declare a0 = <p0>
  // CHECK-NOT: declare a2 = <2>
  kgen.param.declare a2 = <2>
  // CHECK-not: declare a1 = <add(p0, 1)>
  kgen.param.declare a1 = <add(p0, 1)>
  // CHECK-NEXT: simd<p0, si8>
  "user"() : () -> !pop.simd<a0, si8>
  // CHECK-NEXT: simd<add(p0, 1), si8>
  "user"() : () -> !pop.simd<a1, si8>
  // CHECK-NEXT: simd<2, si8>
  "user"() : () -> !pop.simd<a2, si8>

  // CHECK-NEXT: F1 = <value: simd<2, si8>>() -> !pop.simd<2, si8>
  kgen.param.declare.region F1 = <value: simd<a2, si8>>() -> !pop.simd<a2, si8>{
    // CHECK-NOT: declare b0 = <a1>
    kgen.param.declare b0 = <a1>
    // CHECK-NOT: declare b1
    kgen.param.declare b1 = <add(2, a2)>
    // CHECK-NEXT: simd<2, si8>
    "user"() : () -> !pop.simd<a2, si8>
    // CHECK-NEXT: simd<add(p0, 1), si8>
    "user"() : () -> !pop.simd<b0, si8>
    // CHECK-NEXT: simd<4, si8>
    "user"() : () -> !pop.simd<b1, si8>
    // CHECK-NOT: declare type_change0: simd<2, si8> = <value>
    kgen.param.declare type_change0: simd<a2, si8> = <value>
    // CHECK-NOT: declare type_change1
    kgen.param.declare type_change1: simd<b0, si8> = <rebind(:simd<a2, si8> value)>
    // CHECK-NEXT: simd<2, si8> = <value>
    %0 = kgen.param.constant: simd<a2, si8> = <type_change0>
    // CHECK-NEXT: simd<add(p0, 1), si8> = <rebind(:simd<2, si8> value)>
    %1 = kgen.param.constant: simd<b0, si8> = <type_change1>
    "user"(%1) : (!pop.simd<b0, si8>) -> ()
    kgen.return %0 : !pop.simd<a2, si8>
  }

  kgen.param.declare dt: dtype = <si8>
  // CHECK: F2 = <c0, c1:  simd<c0, si8>>(%arg0: !pop.simd<c0, si8>
  kgen.param.declare.region F2 = <c0, c1: simd<c0, dt>>(%arg0: !pop.simd<c0, dt>) {
    // CHECK-NEXT: <c0, :dtype si8, :simd<c0, si8> c1>
    kgen.call @callee_simd<c0, :dtype dt, :simd<c0, dt> c1>() : () -> ()
    kgen.return
  }

  // CHECK: :() -> () @unbound_fn<2>
  kgen.call @callee_fn<:() -> () @unbound_fn<a2>>() : () -> ()

  // CHECK: constant = <2>
  kgen.param.constant = <a2>
  kgen.return
}

kgen.generator @result_slot(%arg0: index, %arg1: !kgen.pointer<index> byref_result) -> !kgen.none {
  pop.store %arg0, %arg1 : !kgen.pointer<index>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: @interpret_result_slot
kgen.generator @interpret_result_slot() {
  // CHECK-NEXT: constant = <42>
  kgen.param.constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @result_slot, 42)>
  kgen.return
}

kgen.generator @uninitialized_empty_struct() -> !kgen.struct<(struct<()>)> {
  %0 = pop.stack_allocation 1 x struct<(struct<()>)>
  %1 = pop.load %0 : !kgen.pointer<struct<(struct<()>)>>
  kgen.return %1 : !kgen.struct<(struct<()>)>
}

// CHECK-LABEL: @load_uninitialized_struct
kgen.generator @load_uninitialized_struct() {
  // CHECK-NEXT: struct<(struct<()>)> = <{ { } }>
  kgen.param.constant: struct<(struct<()>)> = <apply(:() -> !kgen.struct<(struct<()>)> @uninitialized_empty_struct)>
  kgen.return
}

// CHECK-LABEL: @duplicate_constraints
kgen.generator @duplicate_constraints<a: i1, b: i1, c: i1, d: i1>() {
  // CHECK-NEXT: assert <a>
  kgen.param.assert <a>, ""
  kgen.param.assert <a>, ""
  // CHECK-NEXT: assert <b>
  kgen.param.assert <b>, ""
  // CHECK-NEXT: param.if
  kgen.param.if <d> {
    kgen.param.assert <a>, ""
    // CHECK-NEXT: assert <c>
    kgen.param.assert <c>, ""
    // CHECK-NEXT: yield
    kgen.param.yield
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: assert <c>
    kgen.param.assert <b>, ""
    kgen.param.assert <c>, ""
    // CHECK-NEXT: yield
    kgen.param.yield
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: param.if
  kgen.param.if <d> {
    kgen.param.assert <a>, ""
    kgen.param.assert <c>, ""
    // CHECK-NEXT: assert <c>
    kgen.param.assert <c>, ""
    // CHECK-NEXT: yield
    kgen.param.yield
  // CHECK-NEXT: else
  } else {
    kgen.param.assert <b>, ""
    // CHECK-NEXT: yield
    kgen.param.yield
  }
  kgen.return
}

// -----

kgen.generator @interpret_me(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @simplify
kgen.generator @simplify() {
  kgen.param.declare x = <2>
  kgen.param.declare y = <apply(:(index) -> index @interpret_me, x)>
  // CHECK-NEXT: constant = <4>
  kgen.param.constant = <y>
  kgen.return
}

kgen.extern.generator @extern() -> index

kgen.generator @extern_apply() {
  kgen.param.constant = <apply(:() -> index @extern)>
  kgen.return
}
