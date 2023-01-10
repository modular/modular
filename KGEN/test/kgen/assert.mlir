// RUN: not not kgen -execute -func="assert_false:()" -o %t.o %s 2>&1 | FileCheck %s

// CHECK: kgen/assert.mlir:8:kgen.func: failed assertion 'assert failure!!!'
kgen.generator @assert_false() {
  %zero = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<0>>
  %one = kgen.param.constant: !pop.scalar<si8> = <#pop.simd<1>>
  %false = pop.cmp eq(%zero, %one) : !pop.simd<1, si8>
  zap.debug_assert %false, "assert failure!!!" : !pop.simd<1, bool>
  kgen.return
}

kgen.export [@assert_false]
