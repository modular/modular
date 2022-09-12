// RUN: kgen-execute %s -execute -func="foo:f32():%t.o" | FileCheck %s

kgen.func @foo() -> f32 {
  %av = pop.constant(1.0 : f32) : !meta.scalar<f32>
  %c10 = pop.constant(10.0 : f32) : !meta.scalar<f32>
  %lb = index.constant 0
  %ub = index.constant 10
  %step = index.constant 1
  %rv = scf.for %i = %lb to %ub step %step iter_args(%v = %av) -> (!meta.scalar<f32>) {
    %n = pop.add %c10, %v : !meta.scalar<f32>
    scf.yield %n : !meta.scalar<f32>
  }
  %r = meta.cast_to_builtin %rv : !meta.scalar<f32> to f32
  kgen.return %r : f32
}

// CHECK: --- 'foo' returned 101.{{[0-9]+}}
