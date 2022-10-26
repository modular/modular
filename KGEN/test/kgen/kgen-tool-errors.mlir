// RUN: kgen %s -execute -func="exp_f32:f32()" -func="exp_f32:f32(f32)" -func="badkernel:f32()" -ignore-failure -I %S/../kernels -verify-diagnostics

// expected-error@-3 {{could not find func '@badkernel'}}

kgen.include "library.mlir"

kgen.generator.interface @exp<type: dtype>(%x: !pop.simd<1, type>) -> !pop.simd<1, type>

// expected-error@below {{command-line specified signature does not match the IR signature}}
kgen.generator public @exp_f32(%arg0: f32) -> (f32, f32) {
  %0 = pop.cast_from_builtin %arg0 : f32 to !pop.simd<1, f32>
  %1 = kgen.call @exp<type: dtype = f32>(%0) : (!pop.simd<1, f32>) -> !pop.simd<1, f32>
  %2 = pop.cast_to_builtin %1 : !pop.simd<1, f32> to f32
  kgen.return %2, %2 : f32, f32
}
