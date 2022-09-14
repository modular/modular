// RUN: kgen-opt %s -elaborate-generators -allow-unregistered-dialect | FileCheck %s

kgen.generator.interface @addOne(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> attributes {
  evalConfigs = #kgen.eval.configurations<[
    #kgen.eval.configuration<random, args=[unit], results=[unit], 256>
  ]>
}

kgen.generator @goodAddOne(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> implements @addOne {
  %0 = pop.constant (1.0 : f32) : !meta.scalar<f32>
  %1 = pop.add %arg0, %0 : !meta.scalar<f32>
  kgen.return %1 : !meta.scalar<f32>
}

kgen.generator @badAddOne(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> implements @addOne {
  %0 = llvm.mlir.constant (1.0 : f32) : f32
  %zero = index.constant 0
  %one = index.constant 1
  %numIters = index.constant 10000
  %acc = meta.cast_to_builtin %arg0: !meta.scalar<f32> to f32
  %tenThousand = scf.for %i = %zero to %numIters step %one iter_args(%a = %acc) -> (f32) {
    %1 = llvm.fadd %a, %0 : f32
    scf.yield %1 : f32
  }
  %out = scf.for %i = %zero to %numIters step %one iter_args(%a = %acc) -> (f32) {
    %1 = llvm.fsub %a, %0 : f32
    scf.yield %1 : f32
  }
  %3 = meta.cast_from_builtin %out: f32 to !meta.scalar<f32>
  %4 = pop.constant (1.0 : f32) : !meta.scalar<f32>
  %5 = pop.add %3, %4 : !meta.scalar<f32>
  kgen.return %5 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.func @returnTwo
kgen.generator @returnTwo() -> !meta.scalar<f32> {
  // CHECK: %[[CST:.*]] = pop.constant
  %0 = pop.constant (1.0 : f32) : !meta.scalar<f32>
  // CHECK: kgen.call @goodAddOne(%[[CST]])
  %1 = kgen.call @addOne(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  kgen.return %1 : !meta.scalar<f32>
}
// CHECK-NOT: kgen.call @badAddOne
