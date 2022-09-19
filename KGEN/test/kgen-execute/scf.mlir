// RUN: kgen-execute %s -execute -func="for_loop:f32():%t.o" | FileCheck %s --check-prefix=FOR
// RUN: kgen-execute %s -execute -func="while_loop:f32():%t.o" | FileCheck %s --check-prefix=WHILE

kgen.func public @for_loop() -> f32 {
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

kgen.func public @while_loop() -> f32 {
  %init = pop.constant(1.2 : f32) : !meta.scalar<f32>
  %limit = pop.constant(10. : f32) : !meta.scalar<f32>
  %result = scf.while (%v = %init) : (!meta.scalar<f32>) -> !meta.scalar<f32> {
    %cmp = pop.cmp lt(%v, %limit) : !meta.scalar<f32>
    %cond = meta.cast_to_builtin %cmp : !meta.scalar<bool> to i1
    scf.condition(%cond) %v : !meta.scalar<f32>
  } do {
  ^bb0(%u : !meta.scalar<f32>):
    %next = pop.mul %u, %u : !meta.scalar<f32>
    scf.yield %next : !meta.scalar<f32>
  }
  %res = meta.cast_to_builtin %result : !meta.scalar<f32> to f32
  kgen.return %res : f32
}

// FOR: --- 'for_loop' returned 101.{{[0-9]+}}
// WHILE: --- 'while_loop' returned 18.4{{[0-9]+}}
