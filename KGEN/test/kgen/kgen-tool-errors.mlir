// RUN: kgen %s -execute -func="exp_f32:f32():%t_exp_f32_bad.o" -func="exp_f32:f32(f32):%t_exp_f32_good.o" -func="badkernel:f32():%t_badkernel.o" -ignore-failure -I %S/../kernels -verify-diagnostics

// expected-error@-3 {{could not find func '@badkernel'}}

kgen.include "library.mlir"

kgen.generator.interface @exp<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>

// expected-error@below {{command-line specified signature does not match the IR signature}}
kgen.generator public @exp_f32(%arg0: f32) -> f32 {
  %0 = pop.type_raise %arg0 : f32 to !meta.scalar<f32>
  %1 = kgen.call @exp<type: dtype = f32>(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  %2 = pop.type_lower %1 : !meta.scalar<f32> to f32
  kgen.return %2 : f32
}
