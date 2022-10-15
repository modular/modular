// RUN: kgen-opt %s -elaborate-generators -allow-unregistered-dialect | FileCheck %s

kgen.generator.interface @addOne(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> attributes {
  evalConfigs = #kgen.eval.configurations<[
    #kgen.eval.configuration<random, args=[unit], results=[unit], 256>
  ]>
}

kgen.generator @goodAddOne(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> implements @addOne {
  %0 = pop.constant (1.0 : f32) : !pop.scalar<f32>
  %1 = pop.add %arg0, %0 : !pop.scalar<f32>
  kgen.return %1 : !pop.scalar<f32>
}

kgen.generator @badAddOne(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> implements @addOne {
  %0 = llvm.mlir.constant (1.0 : f32) : f32
  %zero = index.constant 0
  %one = index.constant 1
  %numIters = index.constant 10000
  %acc = pop.cast_to_builtin %arg0: !pop.scalar<f32> to f32
  %tenThousand = scf.for %i = %zero to %numIters step %one iter_args(%a = %acc) -> (f32) {
    %1 = llvm.fadd %a, %0 : f32
    scf.yield %1 : f32
  }
  %out = scf.for %i = %zero to %numIters step %one iter_args(%a = %acc) -> (f32) {
    %1 = llvm.fsub %a, %0 : f32
    scf.yield %1 : f32
  }
  %3 = pop.cast_from_builtin %out: f32 to !pop.scalar<f32>
  %4 = pop.constant (1.0 : f32) : !pop.scalar<f32>
  %5 = pop.add %3, %4 : !pop.scalar<f32>
  kgen.return %5 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func public @returnTwo
kgen.generator public @returnTwo() -> !pop.scalar<f32> {
  // CHECK: %[[CST:.*]] = pop.constant
  %0 = pop.constant (1.0 : f32) : !pop.scalar<f32>
  // CHECK: kgen.call @goodAddOne(%[[CST]])
  %1 = kgen.call @addOne(%0) : (!pop.scalar<f32>) -> !pop.scalar<f32>
  kgen.return %1 : !pop.scalar<f32>
}
// CHECK-NOT: kgen.call @badAddOne


// CHECK-LABEL: kgen.func public @"even_only,param=72"()
// CHECK-NOT: @"even_only,
// CHECK-LABEL: kgen.func public @"even_only,param=16"() {
// CHECK-NOT: @"even_only,
kgen.generator public @even_only<param>() {
  kgen.param.assert <eq(and(param, 1), 0)>, "the param shalt be even!"
  kgen.return
}

// This should turn into two variants exactly, not a duplicate for 72.
// CHECK-LABEL: kgen.func public @find_even
// CHECK-LABEL: kgen.func public @find_even
// CHECK-NOT: kgen.func public @find_even
kgen.generator public @find_even() {
  kgen.param.search seventy_two = <72>
  kgen.param.search value = <3, 16, 1, 72, seventy_two>
  kgen.call @even_only<param=value>() : () -> ()
  kgen.return
}
