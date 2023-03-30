// RUN: kgen-opt %s -verify-parameters | FileCheck %s

// CHECK-LABEL: kgen.generator @parameterIsolatedRegions
kgen.generator @parameterIsolatedRegions<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region Fn = <B>() {
    kgen.param.constant = <B>
    kgen.return
  }
  // CHECK: {isolated}

  // CHECK: kgen.param.if
  kgen.param.if <lt(A, 1)> {
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  // CHECK: {elseIsolated, thenIsolated}
  kgen.return
}

// CHECK-LABEL: kgen.generator @struct_of_simd
// CHECK-SAME: -> !pop.struct<simd<size, type>>
kgen.generator @struct_of_simd<size, type: dtype>(%arg0: !pop.simd<size, type>) -> !pop.struct<simd<size, type>> {
  %1 = pop.struct.create(%arg0) : !pop.struct<simd<size, type>>
  kgen.return %1 : !pop.struct<simd<size, type>>
}

// CHECK-LABEL: kgen.generator @call_it
kgen.generator @call_it<size, type: dtype, target: dtype>(%arg0: !pop.struct<simd<size, type>>) -> !pop.struct<simd<size, target>> {
  %1 = pop.struct.extract %arg0[0] : !pop.struct<simd<size, type>>
  %3 = pop.cast %1 : !pop.simd<size, type> to !pop.simd<size, target>
  // CHECK: kgen.call @struct_of_simd<size = size, type: dtype = target>
  // CHECK-SAME: (!pop.simd<size, target>) -> !pop.struct<simd<size, target>>
  %4 = kgen.call @struct_of_simd<size = size, type: dtype = target>(%3) : (!pop.simd<size, target>) -> !pop.struct<simd<size, target>>
  kgen.return %4 : !pop.struct<simd<size, target>>
}
