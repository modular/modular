// RUN: not not kgen -execute -func="assert_false:():%t.o" %s 2>&1 | FileCheck %s

// CHECK: kgen/assert.mlir:6:kgen.func: failed assertion 'assert failure!!!'
kgen.generator public @assert_false() {
  %cond = pop.constant(false) : !pop.scalar<bool>
  zap.debug_assert %cond, "assert failure!!!" : !pop.scalar<bool>
  kgen.return
}
