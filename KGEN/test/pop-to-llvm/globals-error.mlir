// RUN: kgen-opt -lower-global-pop-to-llvm -verify-diagnostics %s

kgen.func @external_call(%a: !pop.scalar<ui32>) {
  // expected-note @below {{see function declaration here}}
  pop.external_call @foo(%a) : (!pop.scalar<ui32>) -> ()
  // expected-error @below {{existing function with conflicting signature}}
  // expected-error @below {{failed to legalize}}
  %0 = pop.external_call @foo(%a) : (!pop.scalar<ui32>) -> !meta.simd<4, f64>
  kgen.return
}
