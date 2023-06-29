// RUN: kgen-opt %s -allow-unregistered-dialect -verify-parameters=simplify=true -verify-parameters

kgen.generator @unbound_fn<p0>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @param_prop
kgen.generator @param_prop<p0, p1 -> p2>() {
  // CHECK-NEXT: declare a0 = <p0>
  kgen.param.declare a0 = <p0>
  // CHECK-NOT: declare a2 = <2>
  kgen.param.declare a2 = <2>
  // CHECK-NEXT: declare a1 = <add(p0, 1)>
  kgen.param.declare a1 = <add(p0, 1)>
  // CHECK-NEXT: simd<a0, si8>
  "user"() : () -> !pop.simd<a0, si8>
  // CHECK-NEXT: simd<a1, si8>
  "user"() : () -> !pop.simd<a1, si8>
  // CHECK-NEXT: simd<2, si8>
  "user"() : () -> !pop.simd<a2, si8>

  // CHECK-NEXT: F1 = <value: simd<2, si8>>() -> !pop.simd<2, si8>
  kgen.param.declare.region F1 = <value: simd<a2, si8>>() -> !pop.simd<a2, si8>{
    // CHECK-NEXT: declare b0 = <a1>
    kgen.param.declare b0 = <a1>
    // CHECK-NOT: declare b1
    kgen.param.declare b1 = <add(2, a2)>
    // CHECK-NEXT: simd<2, si8>
    "user"() : () -> !pop.simd<a2, si8>
    // CHECK-NEXT: simd<b0, si8>
    "user"() : () -> !pop.simd<b0, si8>
    // CHECK-NEXT: simd<4, si8>
    "user"() : () -> !pop.simd<b1, si8>
    // CHECK-NEXT: declare type_change0: simd<2, si8> = <value>
    kgen.param.declare type_change0: simd<a2, si8> = <value>
    // CHECK-NEXT: declare type_change1: simd<b0, si8> = <rebind(:simd<2, si8> value)>
    kgen.param.declare type_change1: simd<b0, si8> = <rebind(:simd<a2, si8> value)>
    // CHECK-MEXT: simd<2, si8> = <type_change0>
    %0 = kgen.param.constant: simd<a2, si8> = <type_change0>
    // CHECK-MEXT: simd<b0, si8> = <type_change1>
    %1 = kgen.param.constant: simd<b0, si8> = <type_change1>
    "user"(%1) : (!pop.simd<b0, si8>) -> ()
    kgen.return %0 : !pop.simd<a2, si8>
  }

  kgen.param.declare dt: dtype = <si8>
  // CHECK: F2 = <c0, c1:  simd<c0, si8>>(%arg0: !pop.simd<c0, si8>)
  kgen.param.declare.region F2 = <c0, c1: simd<c0, dt>>(%arg0: !pop.simd<c0, dt>) {
    // CHECK-NEXT: fork f0: simd<c0, si8> = <[c1]>
    kgen.param.fork f0: simd<c0, dt> = <[c1]>
    kgen.return
  }

  // CHECK: bound: () -> () = <[@unbound_fn<2>]>
  kgen.param.fork bound: () -> () = <[@unbound_fn<a2>]>

  // CHECK: result_bind<2>
  kgen.param.result_bind<a2>
  kgen.return
}
